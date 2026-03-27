from openai import OpenAI
import openai  
import time
import os
from .utils import deal_answer, deal_judge_new, has_answer
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

client = OpenAI(
    api_key="",   # API KEY
    base_url="",  # API SOURCE
)

def get_res_from_chat(messages, args):
    """
    发送单条消息给 OpenAI API，并返回结果（字符串）。
    v1.x: 使用 client.chat.completions.create，返回对象属性访问
    """
    max_tokens = 2048
    while True:
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=max_tokens,
                # v1.x 支持 chat logprobs（若不需要可去掉）
                logprobs=True
            )
            text = resp.choices[0].message.content or ""
            return text.strip()

        except (openai.RateLimitError,
                openai.APIConnectionError,
                openai.APIError,
                openai.AuthenticationError,
                openai.BadRequestError) as e:
            print(f'\n{type(e).__name__}\t{e}\tRetrying...')
            time.sleep(5)

        except Exception as e:
            # 其他未知异常：打印并返回空串，避免上游写出 null
            print(e)
            return ""

def normalize_judgement(text: str) -> str:
    """
    规范化模型返回，只允许 'certain' 或 'uncertain'（不区分大小写）。
    其他任何输出都视为 'uncertain'。
    """
    if not isinstance(text, str):
        return "uncertain"
    t = text.strip().lower()
    if t == "certain":
        return "certain"
    if t == "uncertain":
        return "uncertain"
    # 可能模型带了多余内容，尝试在前缀中抓取
    if "certain" in t and "uncertain" not in t:
        return "certain"
    if "uncertain" in t:
        return "uncertain"
    return "uncertain"


def get_llm_result(prompts, samples, args):
    """
    支持批量并发调用 API，并保证输出顺序与输入一致。
    返回：与输入顺序一致的列表；每个元素都是 gold 格式的 dict：
    {
      'question': <from sample>,
      'prompt': <the prompt we actually sent>,
      'answer': <from sample>,
      'response': [<RAW model output string>]
    }
    """
    results = [None] * len(prompts)  # 预分配，保证顺序

    def process_one(index: int, prompt: str, sample: dict):
        """
        调用一次模型，并封装为 gold 格式。
        - prompt: 直接使用上游 prepare_prompts(args.type) 生成的 instruction（可含 <|im_start|> 等）
        - sample: 需要至少包含 question / answer
        """
        messages = [{"role": "user", "content": prompt}]
        model_raw = get_res_from_chat(messages, args)  # 原始字符串（可能是 judgement 或长回答）
        out = {
            "question": sample.get("question", ""),
            "prompt": prompt,
            "answer": sample.get("answer", []),
            "response": [model_raw],   # 不再归一化为 certain/uncertain，直接原样放入列表
        }
        return index, out

    # 线程池并发
    max_workers = max(1, int(getattr(args, "batch_size", 5)))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_one, i, prompts[i], samples[i]): i
            for i in range(len(prompts))
        }

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                index, out = future.result()
                results[index] = out
            except Exception as e:
                print(f"Error processing prompt at index {i}: {e}")
                # 兜底：保持 gold 结构，response 放空串以避免 null
                results[i] = {
                    "question": samples[i].get("question", ""),
                    "prompt": prompts[i],
                    "answer": samples[i].get("answer", []),
                    "response": [""],
                }

    return results
