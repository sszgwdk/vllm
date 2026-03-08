# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with and without prefix caching using either fixed prompts
or prompts sampled from the ShareGPT dataset.

Fixed example usage:
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100 \
        --input-length-range 128:256

ShareGPT example usage:
    # This command samples 20 prompts with input lengths
    # between 128 and 256 tokens from the ShareGPT dataset,
    # then replicates each prompt 5 times.
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-prefix-caching \
        --num-prompts 20 \
        --repeat-count 5 \
        --input-length-range 128:256
"""

import dataclasses
import json
import math
import random
import time
from typing import Optional

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501


def get_detailed_metrics(llm):
    metrics = {}
    try:
        from vllm.v1.metrics.loggers import LoggingStatLogger
        if hasattr(llm.llm_engine, 'logger_manager'):
            logger_manager = llm.llm_engine.logger_manager
            if logger_manager is None:
                return None
            
            for engine_loggers in logger_manager.per_engine_logger_dict.values():
                for logger in engine_loggers:
                    if isinstance(logger, LoggingStatLogger):
                        metrics['prefix_cache_hit_rate'] = logger.prefix_caching_metrics.hit_rate
                        if hasattr(logger.prefix_caching_metrics, 'total_hit_rate'):
                            metrics['prefix_cache_total_hit_rate'] = logger.prefix_caching_metrics.total_hit_rate

                        # Detailed hit rates by source (GPU vs connector)
                        for attr in [
                            ('gpu_prefix_cache_hit_rate', 'gpu_hit_rate'),
                            ('connector_prefix_cache_hit_rate', 'connector_hit_rate'),
                            ('gpu_prefix_cache_total_hit_rate', 'total_gpu_hit_rate'),
                            ('connector_prefix_cache_total_hit_rate',
                             'total_connector_hit_rate'),
                        ]:
                            key, prop = attr
                            if hasattr(logger.prefix_caching_metrics, prop):
                                metrics[key] = getattr(logger.prefix_caching_metrics, prop)
                        
                        if hasattr(logger, 'cumulative_prompt_tokens'):
                            metrics['total_prompt_tokens'] = logger.cumulative_prompt_tokens
                        if hasattr(logger, 'cumulative_generation_tokens'):
                            metrics['total_generation_tokens'] = logger.cumulative_generation_tokens
                        
                        return metrics
    except Exception:
        pass
    return None


def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()

    # Generate with return_outputs=True to capture the outputs
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    end_time = time.time()
    duration = end_time - start_time
    print(f"cost time {duration:.2f}s")
    
    # Print the first few outputs to verify correctness
    print("\n--- Sample Outputs (first 3) ---")
    for i, output in enumerate(outputs[:3]):  # Print first 3 outputs
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i+1}: {prompt[:100]}...")
        print(f"Generated {i+1}: {generated_text}")
        print("---")
    
    # Print total number of outputs generated
    print(f"\nTotal prompts processed: {len(outputs)}")
    
    metrics = get_detailed_metrics(llm)
    if metrics:
        if 'prefix_cache_total_hit_rate' in metrics:
            print(f"Prefix cache hit rate (total): {metrics['prefix_cache_total_hit_rate'] * 100:.2f}%")
            if 'gpu_prefix_cache_total_hit_rate' in metrics:
                print(f"  GPU hit rate (total): {metrics['gpu_prefix_cache_total_hit_rate'] * 100:.2f}%")
            if 'connector_prefix_cache_total_hit_rate' in metrics:
                print(f"  Connector hit rate (total): {metrics['connector_prefix_cache_total_hit_rate'] * 100:.2f}%")
        
        if 'total_prompt_tokens' in metrics and 'total_generation_tokens' in metrics:
            total_tokens = metrics['total_prompt_tokens'] + metrics['total_generation_tokens']
            print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
            print(f"Total generation tokens: {metrics['total_generation_tokens']}")
            print(f"Total tokens: {total_tokens}")
            print(f"Average throughput: {total_tokens / duration:.2f} tokens/s")


@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int


def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)

    # Remove the special tokens.
    return random.choices(
        [v for k, v in vocab.items() if k not in all_special_ids],
        k=length,
    )


def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: Optional[int],
    seed: Optional[int] = None,
) -> list[Request]:
    # if fixed_output_len is not None and fixed_output_len < 4:
    #     raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    if seed is not None:
        random.Random(seed).shuffle(dataset)
    else:
        random.shuffle(dataset)

    min_len, max_len = input_length_range
    assert min_len >= 0 and max_len >= min_len, "input_length_range too small"

    # Filter out sequences that are too long or too short
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if min_len <= prompt_len <= max_len:
            filtered_requests.append(Request(prompt, prompt_len, output_len))

    return filtered_requests


def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: Optional[int],
    prefix_len: int,
) -> list[Request]:
    requests = []
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range

    for i in range(num_requests):
        unique_part_token_ids = sample_tokens(
            tokenizer, random.randint(min_len - prefix_len, max_len - prefix_len)
        )
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)
        assert min_len <= prompt_len <= max_len, (
            f"prompt_len {prompt_len} out of range {min_len}:{max_len}"
        )
        requests.append(Request(prompt, prompt_len, fixed_output_len))
    return requests


def repeat_and_sort_requests(
    requests: list[Request], repeat_count: int, sort: bool = False, seed: Optional[int] = None
) -> list[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda r: r.prompt_len)
    else:
        if seed is not None:
            random.Random(seed).shuffle(repeated_requests)
        else:
            random.shuffle(repeated_requests)
    return [req.prompt for req in repeated_requests]


def zipf_repeat_requests(
    requests: list[Request],
    total_count: int,
    exponent: float = 0.8,
    sort: bool = False,
    seed: Optional[int] = None,
) -> list[str]:
    """Repeat requests according to a Zipf distribution over ranks.

    Guarantees each request appears at least once.

    Args:
        requests: Base unique requests.
        total_count: Total number of requests to generate.
        exponent: Zipf exponent s.
        sort: If True, sort by prompt length (frequency unchanged).
        seed: RNG seed for reproducibility.
    """
    if not requests:
        return []

    rng = random.Random(seed)

    # Randomly assign popularity ranks under the given seed.
    ranked_requests: list[Request] = list(requests)
    rng.shuffle(ranked_requests)
    n = len(ranked_requests)

    # Ensure each prompt appears at least once.
    total_count = max(total_count, n)

    # Assign popularity by rank (1..n). rank=1 is the most popular.
    weights = [1.0 / math.pow(rank, exponent) for rank in range(1, n + 1)]
    remaining = total_count - n

    # Ensure each prompt appears at least once.
    repeated: list[Request] = list(ranked_requests)
    if remaining > 0:
        # Sample additional requests with replacement according to Zipf weights.
        sampled_indices = rng.choices(range(n), weights=weights, k=remaining)
        repeated.extend(ranked_requests[i] for i in sampled_indices)

    if sort:
        repeated.sort(key=lambda r: r.prompt_len)
    else:
        rng.shuffle(repeated)
    return [r.prompt for r in repeated]


def main(args):
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(":")))
    random.seed(args.seed)
    if args.dataset_path is not None:
        if args.prefix_len > 0:
            raise ValueError(
                "prefix-len is not supported when dataset-path is provided."
            )
        print(f"Start to sample {args.num_prompts} prompts from {args.dataset_path}")
        filtered_requests = sample_requests_from_dataset(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
            seed=args.shuffle_seed,
        )
    else:
        print(f"Start to sample {args.num_prompts} prompts from random")
        filtered_requests = sample_requests_from_random(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
            prefix_len=args.prefix_len,
        )

    # Print some helpful stats of the requests.
    print(f"Sampled {len(filtered_requests)} requests.")
    prompt_lens = [req.prompt_len for req in filtered_requests]
    print(f"Average input length: {sum(prompt_lens) / len(prompt_lens)}")
    print(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}")
    print(f"Min Prompt Length: {min(prompt_lens)}")
    print(f"Max Prompt Length: {max(prompt_lens)}")

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.disable_log_stats = False
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    print("Testing filtered requests")
    if args.use_zipf:
        if args.zipf_scale < 1:
            raise ValueError("--zipf-scale must be >= 1")

        # In Zipf mode, the total request volume is controlled by zipf_scale.
        # Note: --repeat-count is ignored in this mode.
        total_count = len(filtered_requests) * args.zipf_scale
        prompts = zipf_repeat_requests(
            filtered_requests,
            total_count=total_count,
            exponent=0.8,
            sort=args.sort,
            seed=args.shuffle_seed,
        )
        print(
            "Using Zipf(s=0.8) frequency over prompts; "
            "each prompt appears at least once. "
            "Note: --repeat-count is ignored in this mode; use --zipf-scale instead."
        )
        print(f"Total requests: {len(prompts)} (unique: {len(filtered_requests)})")
    else:
        prompts = repeat_and_sort_requests(
            filtered_requests,
            repeat_count=args.repeat_count,
            sort=args.sort,
            seed=args.shuffle_seed,
        )

    print("------start generating------")
    test_prefix(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the performance with or without "
        "automatic prefix caching."
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument("--output-len", type=int, default=10)
    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of the prompts sampled from dataset",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Number of times to repeat each prompt (ignored when --use-zipf is set)",
    )
    parser.add_argument(
        "--use-zipf",
        action="store_true",
        help=(
            "Use Zipf(s=0.8) frequency distribution over sampled prompts "
            "to simulate a skewed query workload. Each prompt appears at least once. "
            "In this mode, --repeat-count is ignored; "
            "use --zipf-scale to control total request count (num_prompts * zipf_scale)."
        ),
    )
    parser.add_argument(
        "--zipf-scale",
        type=int,
        default=3,
        help=(
            "Zipf mode only: total request multiplier. "
            "Total requests = num_prompts * zipf_scale (must be >= 1)."
        ),
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort prompts by input length"
    )
    parser.add_argument(
        "--input-length-range",
        type=str,
        required=True,
        help="Range of input lengths for sampling prompts,"
        'specified as "min:max" (e.g., "128:256").',
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Specifies the length of a common prefix to be "
        "added to the input prompt. The input-length-range will "
        "subtract this length when filtering prompts. Only used "
        "when dataset-path is not provided.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=1,
        help="Seed for shuffling the dataset",
    )

    parser = EngineArgs.add_cli_args(parser)

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)