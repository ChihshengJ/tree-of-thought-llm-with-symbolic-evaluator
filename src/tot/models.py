import os
import openai
import backoff
import time
from openai import OpenAIError, RateLimitError, APIError  # Import the OpenAIError exception
from groq import Groq
import anthropic
import google.generativeai as genai


completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base
    
    
groq_api_key = os.getenv("GROQ_API_KEY", "")
if groq_api_key != "":
    groq_client = Groq(api_key=groq_api_key)
else:
    print("Warning: GROQ_API_KEY is not set")
    
    
try:
    genai.configure(api_key=os.environ["API_KEY"])
except Exception:
    print(f"Warning: genimi API_KEY is not set") 
    
    
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
if anthropic_api_key != "":
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
else:
    print("Warning: Anthropic_API_KEY is not set")


@backoff.on_exception(backoff.expo,
                        (OpenAIError, RateLimitError, APIError),
                        max_tries=7)
def completions_with_backoff(**kwargs):
    if kwargs["model"] == "gpt-3.5-turbo" or kwargs["model"] == "gpt-4":
        try:
            return openai.chat.completions.create(model=kwargs["model"],
                                                    messages=kwargs["messages"],
                                                    temperature=kwargs["temperature"],
                                                    max_tokens=kwargs["max_tokens"],
                                                    n=kwargs["n"],
                                                    stop=kwargs["stop"],)
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            raise 
        except APIError as e:
            print(f"API error occurred: {e}")
            raise 
        except OpenAIError as e:
            print(f"General OpenAI error: {e}")
            raise 
    elif kwargs["model"] == "llama-3.1-8b-instant" or kwargs["model"] == "llama-3.1-70b-versatile":
        return groq_client.chat.completions.create(model=kwargs["model"],
                                                    messages=kwargs["messages"],
                                                    temperature=kwargs["temperature"],
                                                    max_tokens=kwargs["max_tokens"],
                                                    n=kwargs["n"],
                                                    stop=kwargs["stop"],)
    elif kwargs["model"] == "claude-3-5-sonnet-20240620" or kwargs["model"] == "claude-3-sonnet-20240229":
        return anthropic_client.messages.create(model=kwargs["model"],
                                                    messages=kwargs["messages"],
                                                    system=kwargs["system"],
                                                    temperature=kwargs["temperature"],
                                                    max_tokens=kwargs["max_tokens"],
                                                    stop_sequences=kwargs["stop"])    
    elif kwargs["model"] == "gemini-1.5-flash" or kwargs["model"] == "gemini-1.5-pro":
        return kwargs["gemini"].generate_content(kwargs["messages"],
                                        generation_config=genai.types.GenerationConfig(
                                            candidate_count=1,
                                            stop_sequences=kwargs["stop"],
                                            max_output_tokens=kwargs["max_tokens"],
                                            temperature=kwargs["temperature"])
                                    )


def gpt(prompt, system_prompt="", model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
    if model == "gpt-3.5-turbo" or model == "gpt-4":
        try:
            return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
        except RateLimitError:
            print("Stopping further execution due to rate limit errors.")
            return [] 
        except Exception as e:
            print(f"An unexpected error occurred when calling a LM: {e}")
            return [] 
        
    elif model == "llama-3.1-8b-instant" or model == "llama-3.1-70b-versatile":
        try:
            return llama(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
        except Exception as e:
            print(f"An unexpected error occurred when calling a LM: {e}")
            return [] 
        
    elif model == "claude-3-5-sonnet-20240620" or model == "claude-3-sonnet-20240229":
        claude_messages = messages[1:]
        system_prompt = messages[0]["content"]
        try:
            return claude(claude_messages, system=system_prompt, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
        except Exception as e:
            print(f"An unexpected error occurred when calling a LM: {e}")
            return []     
        
    elif model == "gemini-1.5-flash" or model == "gemini-1.5-pro":
        gemini_message = messages[1]["content"]
        system_prompt = messages[0]["content"]
        try:
            return gemini(gemini_message, system=system_prompt, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
        except Exception as e:
            print(f"An unexpected error occurred when calling a LM: {e}")
            return []  

def chatgpt(messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
        )
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs


def claude(messages, system="", model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    request_interval = 0.1
    
    while n > 0:
        try:
            res = completions_with_backoff(
                model=model,
                messages=messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=stop,
            )
            outputs.append(res.content[0].text)
            # print("raw_outputs:",outputs)
            completion_tokens += res.usage.output_tokens
            prompt_tokens += res.usage.input_tokens
            n -= 1
            
            if n > 0:
                time.sleep(request_interval)
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred when calling claude: {e}")
            break

    return outputs


def llama(messages, model="llama-3.1-8b-instant", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    request_interval = 0.3 
    
    while n > 0:
        try:
            res = completions_with_backoff(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,  #Groq API doesn't support higher values
                stop=stop,
            )
            outputs.append(res.choices[0].message.content)
            completion_tokens += res.usage.completion_tokens
            prompt_tokens += res.usage.prompt_tokens
            n -= 1
            
            if n > 0:
                time.sleep(request_interval)
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred when calling llama: {e}")
            break

    return outputs


def gemini(messages, system="", model="gemini-1.5-flash", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    request_interval = 0.1
    gemini_instance = genai.GenerativeModel(model_name=model, system_instruction=system)
    while n > 0:
        try:
            res = completions_with_backoff(
                model=model,
                gemini=gemini_instance,
                messages=messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=stop,
            )
            outputs.append(res.text)
            # print("raw_outputs:",outputs)
            completion_tokens += res.usage_metadata.candidates_token_count
            prompt_tokens += res.usage_metadata.prompt_token_count
            n -= 1
            
            if n > 0:
                time.sleep(request_interval)
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred when calling gemini: {e}")
            break

    return outputs


def gpt_usage(backend="gpt-3.5-turbo"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    else:
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}