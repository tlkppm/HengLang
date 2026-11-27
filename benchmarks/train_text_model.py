from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "text_model.heng"
STEMS = ("the fox rests near ", "the owl waits near ", "the hare hides near ", "the deer walks near ")
PLACES = ("old oaks", "still ponds", "stone walls", "quiet paths")
PROMPTS = ("the fox ", "the owl ", "the hare ")
SEQUENCE_END = "\n"
MODULUS = 2147483647
MULTIPLIER = 48271
DEFAULT_EPOCHS = 36
EVALUATION_SPLITS = ("train", "validation", "validation_unseen")


class ExperimentError(RuntimeError):
    pass


class ExperimentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ExperimentError("命令行参数无效，请使用 --help 查看允许的参数")


def parse_arguments(arguments):
    parser = ExperimentParser(prog="train_text_model", description="使用合成语料训练字符级文本模型并输出 JSON")
    extent = parser.add_mutually_exclusive_group()
    extent.add_argument("--epochs", type=int, help=f"训练轮数，默认 {DEFAULT_EPOCHS}，允许 1 至 100")
    extent.add_argument("--smoke", action="store_true", help="执行一轮训练和生成，不调用训练计时器")
    parser.add_argument("--numpy", action="store_true", help="在 HENG 之后串行运行 NumPy 手写基线")
    parser.add_argument("--require-win", action="store_true", help="要求模型指标与续写一致，且 HENG 在核实单线程后至少快 10%%")
    parser.add_argument("--minimum-speedup", type=float, default=1.10, help="严格训练验收要求的 NumPy/HENG 最低速度比，默认 1.10")
    parser.add_argument("--native-step", action="store_true", help="使用字符模型专用原生训练步，保留通用自动微分默认路径")
    parser.add_argument("--heng-matmul", choices=("default", "strict", "fast"), default="default",
                        help="strict 或 fast 通过 set_matmul_fast 显式切换路径，默认沿用运行时设置")
    parser.add_argument("--timeout", type=float, default=120.0, help="HENG 程序执行超时秒数")
    parser.add_argument("--build-timeout", type=float, default=300.0, help="HENG 构建与执行的总超时秒数")
    args = parser.parse_args(arguments)
    if args.require_win and (not args.numpy or args.smoke):
        raise ExperimentError("--require-win 需要 NumPy 对照和正式训练计时")
    if args.epochs is not None and not 1 <= args.epochs <= 100:
        raise ExperimentError("训练轮数必须在 1 至 100 之间")
    if not 0 < args.timeout <= args.build_timeout < math.inf or not math.isfinite(args.minimum_speedup) or args.minimum_speedup <= 0:
        raise ExperimentError("超时和最低速度比必须为有限正数，且总超时不得小于执行超时")
    return args


def corpus():
    train, validation = [], []
    for stem_index, stem in enumerate(STEMS):
        for place_index, place in enumerate(PLACES):
            split = validation if stem_index == place_index else train
            split.append(stem + place + ".")
    return train, validation


def model_configuration(vocabulary, args):
    return {
        "vocabulary": vocabulary,
        "context": 12,
        "hidden": 64,
        "batch_size": 32,
        "epochs": 1 if args.smoke else (args.epochs or DEFAULT_EPOCHS),
        "seed": 1729,
        "shuffle_seed": 2718,
        "input_scale": 0.25,
        "output_scale": 0.125,
        "learning_rate": 0.02,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 0.00000001,
        "generation_length": 48,
        "timed": not args.smoke,
        "native_step": args.native_step,
    }


def encode_split(sentences, character_ids):
    return [character_ids[character] for sentence in sentences for character in sentence + SEQUENCE_END]


def encode_prompt(prompt, character_ids, width):
    context = [character_ids[character] for character in prompt[-width:]]
    return [0] * (width - len(context)) + context


def heng_source(train_tokens, validation_tokens, unseen_indices, prompts, config, matmul):
    fields = ", ".join(f"{key}: {str(value).lower() if isinstance(value, bool) else value}"
                       for key, value in config.items())
    prompt_tokens = [token for prompt in prompts for token in prompt]
    matmul_setting = {"default": "", "strict": "set_matmul_fast(false);", "fast": "set_matmul_fast(true);"}[matmul]
    return EXAMPLE.read_text(encoding="utf-8") + f"""
fn main() {{
    set_threads(1);
    set_optimized(true);
    {matmul_setting}
    let config = TextModelConfig {{ {fields} }};
    text_model_run({train_tokens}, {validation_tokens}, {unseen_indices}, {prompt_tokens}, config);
}}
"""


def metrics(loss, accuracy):
    if not math.isfinite(loss) or loss < 0 or not 0 <= accuracy <= 1:
        raise ExperimentError("模型返回了无效的损失或准确率")
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": accuracy}


def decode_generation(prompt, ids, vocabulary, limit):
    if (not isinstance(ids, list) or not 1 <= len(ids) <= limit
            or any(type(token) is not int or not 0 <= token < len(vocabulary) for token in ids)
            or 0 in ids[:-1] or (ids[-1] != 0 and len(ids) != limit)):
        raise ExperimentError("模型返回了无效的生成序列")
    ended = ids[-1] == 0
    continuation = "".join(vocabulary[token] for token in (ids[:-1] if ended else ids))
    return {"prompt": prompt, "continuation": continuation, "text": prompt + continuation,
            "stopped_at_sentence_end": ended}


def parse_heng_output(output, vocabulary, config):
    lines = output.splitlines()
    evaluation_end = 2 + 4 * len(EVALUATION_SPLITS)
    if (len(lines) != evaluation_end + 2 + len(PROMPTS) or lines[0] != "HENG_TEXT_MODEL_V2"
            or lines[1] != "1" or lines[-1] != "HENG_TEXT_MODEL_END_V1"):
        raise ExperimentError("HENG 文本模型输出协议有误或单线程配置校验失败")
    scores = [metrics(float(lines[index]), float(lines[index + 1])) for index in range(2, evaluation_end, 2)]
    seconds = float(lines[evaluation_end])
    if not (math.isfinite(seconds) and (seconds > 0 if config["timed"] else seconds == 0)):
        raise ExperimentError("HENG 训练计时无效")
    generations = [decode_generation(prompt, json.loads(value), vocabulary, config["generation_length"])
                   for prompt, value in zip(PROMPTS, lines[evaluation_end + 1:-1])]
    return {"initial": dict(zip(EVALUATION_SPLITS, scores[:len(EVALUATION_SPLITS)])),
            "final": dict(zip(EVALUATION_SPLITS, scores[len(EVALUATION_SPLITS):])),
            "training_seconds": seconds if config["timed"] else None,
            "generations": generations, "single_thread_verified": True}


def run_heng(train_tokens, validation_tokens, unseen_indices, prompts, vocabulary, config, args):
    build_directory = ROOT / ".build"
    build_directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="text-model-", dir=build_directory) as directory:
        source = Path(directory) / "experiment.heng"
        source.write_text(heng_source(train_tokens, validation_tokens, unseen_indices, prompts, config, args.heng_matmul),
                          encoding="utf-8", newline="\n")
        command = [sys.executable, "-B", "-m", "henglang", "run", str(source), "--timeout", str(args.timeout)]
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
                                    timeout=args.build_timeout)
        except subprocess.TimeoutExpired:
            raise ExperimentError("HENG 构建和执行超过总超时") from None
    if result.returncode:
        if args.heng_matmul != "default" and "set_matmul_fast" in result.stderr:
            raise ExperimentError("当前 HENG 无法使用 set_matmul_fast，请检查该接口或使用默认矩阵模式")
        raise ExperimentError("HENG 构建或执行失败。原始诊断已隐藏以保护本地信息")
    return parse_heng_output(result.stdout, vocabulary, config)


def initialized_weights(np, rows, columns, seed, scale):
    entries = []
    state = seed
    for _ in range(rows * columns):
        state = (state * MULTIPLIER) % MODULUS
        entries.append((2.0 * (state / MODULUS) - 1.0) * scale)
    return np.asarray(entries, dtype=np.float64).reshape(rows, columns)


def epoch_orders(count, seed):
    state = seed % (MODULUS - 1) + 1
    while True:
        order = list(range(count))
        for index in range(count - 1, 0, -1):
            state = (state * MULTIPLIER) % MODULUS
            position = state % (index + 1)
            order[index], order[position] = order[position], order[index]
        yield order


def token_windows(tokens, width):
    window = (0,) * width
    for token in tokens:
        yield window, token
        window = window[1:] + (token,) if token else (0,) * width


def numpy_dataset(np, tokens, config):
    contexts, targets = zip(*token_windows(tokens, config["context"]))
    one_hot = np.eye(config["vocabulary"], dtype=np.float64)
    return one_hot[np.asarray(contexts)].reshape(len(tokens), -1), np.asarray(targets, dtype=np.int64)


def numpy_forward(np, inputs, parameters):
    first_weights, first_bias, second_weights, second_bias = parameters
    hidden = np.tanh(inputs @ first_weights + first_bias)
    return hidden, hidden @ second_weights + second_bias


def numpy_log_probabilities(np, logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def numpy_metrics(np, dataset, parameters):
    inputs, targets = dataset
    _, logits = numpy_forward(np, inputs, parameters)
    log_probabilities = numpy_log_probabilities(np, logits)
    loss = -log_probabilities[np.arange(len(targets)), targets].mean()
    accuracy = (logits.argmax(axis=1) == targets).mean()
    return metrics(float(loss), float(accuracy))


def numpy_gradients(np, inputs, targets, parameters):
    hidden, logits = numpy_forward(np, inputs, parameters)
    logits_gradient = np.exp(numpy_log_probabilities(np, logits))
    logits_gradient[np.arange(len(targets)), targets] -= 1.0
    logits_gradient /= len(targets)
    hidden_gradient = (logits_gradient @ parameters[2].T) * (1.0 - hidden * hidden)
    return (inputs.T @ hidden_gradient, hidden_gradient.sum(axis=0),
            hidden.T @ logits_gradient, logits_gradient.sum(axis=0))


def numpy_train(np, dataset, parameters, config):
    inputs, targets = dataset
    orders = epoch_orders(len(targets), config["shuffle_seed"])
    order = next(orders)
    first = [np.zeros_like(parameter) for parameter in parameters]
    second = [np.zeros_like(parameter) for parameter in parameters]
    beta1, beta2 = config["beta1"], config["beta2"]
    beta1_power = beta2_power = 1.0
    batch_size = config["batch_size"]
    start = time.perf_counter() if config["timed"] else None
    for epoch in range(config["epochs"]):
        if epoch:
            order = next(orders)
        for offset in range(0, len(targets), batch_size):
            indices = order[offset:offset + batch_size]
            gradients = numpy_gradients(np, inputs[indices], targets[indices], parameters)
            beta1_power *= beta1
            beta2_power *= beta2
            for index, gradient in enumerate(gradients):
                first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
                second[index] = beta2 * second[index] + (1.0 - beta2) * (gradient * gradient)
                first_corrected = first[index] * (1.0 / (1.0 - beta1_power))
                second_corrected = second[index] * (1.0 / (1.0 - beta2_power))
                update = first_corrected / (np.sqrt(second_corrected) + config["epsilon"])
                parameters[index] -= config["learning_rate"] * update
    return time.perf_counter() - start if start is not None else None


def numpy_generate(np, prompt, parameters, config):
    ids = []
    one_hot = np.eye(config["vocabulary"], dtype=np.float64)
    for _ in range(config["generation_length"]):
        _, logits = numpy_forward(np, one_hot[prompt].reshape(1, -1), parameters)
        token = int(logits.argmax(axis=1)[0])
        ids.append(token)
        if token == 0:
            break
        prompt = prompt[1:] + [token]
    return ids


def run_numpy(np, train_tokens, validation_tokens, unseen_indices, prompts, vocabulary, config):
    from benchmarks.compare_frameworks import threadpool_information

    input_width, hidden, output_width = config["context"] * len(vocabulary), config["hidden"], len(vocabulary)
    seed = config["seed"]
    parameters = [initialized_weights(np, input_width, hidden, seed, config["input_scale"]), np.zeros(hidden),
                  initialized_weights(np, hidden, output_width, seed + 1, config["output_scale"]), np.zeros(output_width)]
    datasets = {"train": numpy_dataset(np, train_tokens, config), "validation": numpy_dataset(np, validation_tokens, config)}
    inputs, targets = datasets["validation"]
    datasets["validation_unseen"] = (inputs[unseen_indices], targets[unseen_indices])
    initial = {split: numpy_metrics(np, data, parameters) for split, data in datasets.items()}
    before = threadpool_information()["verified"]
    seconds = numpy_train(np, datasets["train"], parameters, config)
    after = threadpool_information()["verified"]
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ExperimentError("NumPy 训练计时无效")
    return {"initial": initial, "final": {split: numpy_metrics(np, data, parameters) for split, data in datasets.items()},
            "training_seconds": seconds, "generations": [
                decode_generation(prompt, numpy_generate(np, encoded, parameters, config), vocabulary, config["generation_length"])
                for prompt, encoded in zip(PROMPTS, prompts)
            ], "single_thread_verified": before and after}


def compare_results(heng, numpy):
    score_pairs = [(heng[phase][split], numpy[phase][split])
                   for phase in ("initial", "final") for split in EVALUATION_SPLITS]
    loss_error = max(abs(left["loss"] - right["loss"]) for left, right in score_pairs)
    accuracy_error = max(abs(left["accuracy"] - right["accuracy"]) for left, right in score_pairs)
    single_thread = heng["single_thread_verified"] and numpy["single_thread_verified"]
    ratio = None
    if single_thread and heng["training_seconds"] is not None:
        ratio = numpy["training_seconds"] / heng["training_seconds"]
    return {"max_absolute_loss_difference": loss_error, "max_absolute_accuracy_difference": accuracy_error,
            "loss_atol": 1e-8, "accuracy_atol": 1e-12,
            "metrics_match": loss_error <= 1e-8 and accuracy_error <= 1e-12,
            "generations_match": heng["generations"] == numpy["generations"],
            "single_thread_verified": single_thread, "numpy_over_heng_training_time": ratio}


def run(args):
    from benchmarks.compare_frameworks import configure_threads, load_framework

    configure_threads()
    train, validation = corpus()
    vocabulary = SEQUENCE_END + "".join(sorted(set("".join(train))))
    character_ids = {character: index for index, character in enumerate(vocabulary)}
    train_tokens = encode_split(train, character_ids)
    validation_tokens = encode_split(validation, character_ids)
    config = model_configuration(len(vocabulary), args)
    train_examples = set(token_windows(train_tokens, config["context"]))
    unseen_indices = [index for index, example in enumerate(token_windows(validation_tokens, config["context"]))
                      if example not in train_examples]
    if set(train) & set(validation) or not unseen_indices:
        raise ExperimentError("留出集必须包含未见过的完整句子和字符预测样本")
    prompts = [encode_prompt(prompt, character_ids, config["context"]) for prompt in PROMPTS]
    heng = run_heng(train_tokens, validation_tokens, unseen_indices, prompts, vocabulary, config, args)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "privacy": {"machine_information": "omitted"},
        "model": "f64 字符级单隐藏层网络，按位置拼接独热编码，tanh 激活，预测下一个字符",
        "optimizer": "Adam，权重衰减为零，按样本平均交叉熵，每轮确定性洗牌，保留尾批",
        "configuration": {**config, "heng_matmul": args.heng_matmul, "requested_threads": 1,
                          "parameter_count": (config["context"] * len(vocabulary) + 1) * config["hidden"]
                                             + (config["hidden"] + 1) * len(vocabulary),
                          "optimizer_steps": config["epochs"] * math.ceil(len(train_tokens) / config["batch_size"]),
                          "minimum_speedup": args.minimum_speedup},
        "data": {"source": "本地合成，四个句首与四个地点组合，按完整句子留出四个对角组合",
                 "train_sentences": len(train), "validation_sentences": len(validation),
                 "train_targets": len(train_tokens), "validation_targets": len(validation_tokens),
                 "validation_targets_with_unseen_context_target_pair": len(unseen_indices),
                 "validation_text": validation, "vocabulary": vocabulary,
                 "sequence_marker": "换行符用于句首左填充和预测句末，窗口不跨句"},
        "generation": "贪心自回归，每步预测下一字符，遇到句末标记或达到生成长度上限即停止",
        "scope": "留出集检验已见词语的新组合。计时和生成结果适用于此合成语料、模型和训练配置",
        "timing": "单次训练循环的墙钟时间，包含批次提取、前向、反向和更新。构建、数据编码、初始评估、最终评估及生成在计时区间外",
        "results": {"heng": heng},
    }
    if args.numpy:
        numpy = run_numpy(load_framework("numpy"), train_tokens, validation_tokens, unseen_indices, prompts, vocabulary, config)
        report["results"]["numpy"] = numpy
        comparison = compare_results(heng, numpy)
        report["comparison"] = comparison
        if not comparison["metrics_match"]:
            report["status"] = "FAIL"
        if args.require_win:
            ratio = comparison["numpy_over_heng_training_time"]
            passed = comparison["metrics_match"] and comparison["generations_match"] and ratio is not None and ratio > args.minimum_speedup
            report["performance_requirement"] = "PASS" if passed else "FAIL"
            report["status"] = report["performance_requirement"]
    return report


def main(arguments=None):
    from benchmarks.compare_frameworks import BenchmarkError

    try:
        report = run(parse_arguments(arguments))
        output = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    except (ExperimentError, BenchmarkError, OSError, RuntimeError, ValueError, OverflowError, KeyError) as error:
        message = str(error) if isinstance(error, ExperimentError) else "实验执行失败。原始诊断已隐藏以保护本地信息"
        if isinstance(error, BenchmarkError):
            message = "NumPy 无法导入或已提前加载，基线需要独立进程和已安装的 NumPy"
        print(json.dumps({"schema_version": 1, "status": "FAIL", "error": message}, ensure_ascii=False))
        return 1
    print(output)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
