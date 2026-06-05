import os
import regex
import json
import time
import pandas as pd
from anytree import Node
import traceback
import subprocess

from openai import OpenAI, AzureOpenAI
import tiktoken

# Optional provider SDKs --- guarded so importing this module doesn't fail
# when a backend we don't use isn't installed. Original upstream imported all
# of vertexai / anthropic-vertex / google-generativeai unconditionally; in this
# vendored copy we only need the backends we actually drive (claude_code,
# anthropic, openai). The other branches still work if the SDK is installed.
try:
    import vertexai
    from vertexai.generative_models import (
        GenerationConfig,
        GenerativeModel as VertexGenerativeModel,
        HarmBlockThreshold as VertexHarmBlockThreshold,
        HarmCategory as VertexHarmCategory,
        SafetySetting,
    )
    _HAS_VERTEX = True
except ImportError:
    _HAS_VERTEX = False

try:
    from anthropic import AnthropicVertex
    _HAS_ANTHROPIC_VERTEX = True
except ImportError:
    _HAS_ANTHROPIC_VERTEX = False

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

from sklearn import metrics
import numpy as np


class APIClient:
    """
    Prompting for OpenAI, VertexAI, and vLLM.

    Parameters:
    - api: API type (e.g., 'openai', 'vertex', 'vllm')
    - model: Model name

    Methods:
    - estimate_token_count: Estimate token count for a prompt
    - truncating: Truncate document to max tokens
    - iterative_prompt: Prompt API one by one with retries
    - batch_prompt: Batch prompting for vLLM API
    """

    def __init__(self, api, model, host=None):
        self.api = api
        self.model = model
        self.client = None
        # Vendored addition: usage tracking, accumulated across iterative_prompt
        # calls so orchestrate.py / batch_assigner.py can record cost in the
        # SPEC §5.11 meta.json schema.
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "n_calls": 0,
        }

        # Setting API key ----
        if api == "openai":
            self.client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        elif api == "vertex":
            if not _HAS_VERTEX:
                raise ImportError("vertex backend requires google-cloud-aiplatform.")
            vertexai.init(
                project=os.environ["VERTEX_PROJECT"],
                location=os.environ["VERTEX_LOCATION"],
            )
            if model.startswith("gemini"):
                self.model_obj = VertexGenerativeModel(self.model)
        elif api == "ollama":
            self.client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            )
        elif api == "vllm":
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError(
                    "vLLM backend requires the optional 'vllm' extra. "
                    "Install with: pip install 'topicgpt_python[vllm]'"
                ) from e
            self.hf_token = os.environ.get("HF_TOKEN")
            self.llm = LLM(
                self.model,
                download_dir=os.environ.get("HF_HOME", None),
            )
            self.tokenizer = self.llm.get_tokenizer()
        elif api == "gemini":
            if not _HAS_GEMINI:
                raise ImportError("gemini backend requires google-generativeai.")
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model_obj = genai.GenerativeModel(self.model)
        elif api == "azure":
            self.client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),
            api_version = "2024-02-01",
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            )
        elif api == "claude_code":
            # Vendored addition: route to the Claude Code Max subscription
            # via benchmarking.llm_clients.claude_code. Used for topic
            # generation / refinement / correction (Opus 4.7, subscription
            # billing). No SDK client to construct here --- the subprocess
            # wrapper is stateless.
            pass
        elif api == "anthropic":
            # Vendored addition: metered Anthropic SDK (Haiku 4.5 sync calls).
            # Batch-API calls do NOT route through APIClient; see
            # benchmarking/baselines/topicgpt/batch_assigner.py.
            if not _HAS_ANTHROPIC:
                raise ImportError(
                    "anthropic backend requires the 'anthropic' package."
                )
            self.client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        else:
            raise ValueError(
                f"API {api} not supported. Custom implementation required."
            )

    def estimate_token_count(self, prompt: str) -> int:
        """
        Estimating the token count for the prompt with tiktoken

        Parameters:
        - prompt: Prompt text

        Returns:
        - token_count: Estimated token
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")

        token_count = len(enc.encode(prompt))
        return token_count

    def truncating(self, document: str, max_tokens: int) -> str:
        """
        Truncating the document to the max tokens

        Parameters:
        - document: Document text
        - max_tokens: Maximum token count

        Returns:
        - truncated_doc: Truncated document
        """
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            print("Warning: model not found. Using o200k_base encoding.")
            enc = tiktoken.get_encoding("o200k_base")

        tokens = enc.encode(document)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return enc.decode(tokens)

    def iterative_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,  # default value for top_p in openai
        system_message: str = "You are a helpful assistant.",
        num_try: int = 3,
        verbose: bool = False,
    ):
        """
        Prompting API one by one with retries

        Parameters:
        - prompt: Prompt text
        - max_tokens: Maximum token count
        - temperature: Temperature for sampling
        - top_p: Top p value for sampling
        - system_message: System message
        - num_try: Number of retries
        - verbose: Verbose mode

        Returns:
        - response: Response text
        """
        # Formatting prompt
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(num_try):
            try:
                if self.api in ["openai", "azure", "ollama"]:
                    completion_params = {
                        "model": self.model,
                        "messages": message,
                    }
                    # Vendored addition: gpt-5* / o1 / o3 reject (or ignore)
                    # temperature/top_p; pass them only for traditional chat
                    # models. Matches the convention in skills/corpus-tools/
                    # scripts/classify.py.
                    if not any(self.model.startswith(p) for p in ("gpt-5", "o1", "o3")):
                        completion_params["temperature"] = temperature
                        completion_params["top_p"] = top_p
                    completion_params[
                        "max_completion_tokens" if self.api == "openai" else "max_tokens"
                    ] = max_tokens

                    completion = self.client.chat.completions.create(
                        **completion_params
                    )

                    # Vendored addition: usage tracking. OpenAI's
                    # ``prompt_tokens`` is the TOTAL (cached + uncached);
                    # we subtract ``cached_tokens`` so ``input_tokens`` means
                    # the non-cached billable portion --- same semantic the
                    # Anthropic branch uses. Cost math downstream stays
                    # provider-neutral.
                    cached = (
                        getattr(getattr(completion.usage, "prompt_tokens_details", None),
                                "cached_tokens", 0) or 0
                    )
                    self.usage["n_calls"] += 1
                    self.usage["input_tokens"] += max(0, completion.usage.prompt_tokens - cached)
                    self.usage["output_tokens"] += completion.usage.completion_tokens
                    self.usage["cache_read_input_tokens"] += cached
                    # OpenAI doesn't separately bill cache creation, so leave
                    # cache_creation_input_tokens at zero.
                    if verbose:
                        print(
                            f"[openai] in={completion.usage.prompt_tokens} "
                            f"(cached={cached}) out={completion.usage.completion_tokens}"
                        )
                    return completion.choices[0].message.content

                elif self.api == "vertex":
                    if self.model.startswith("claude"):
                        if not _HAS_ANTHROPIC_VERTEX:
                            raise ImportError(
                                "vertex+claude requires the 'anthropic[vertex]' extra."
                            )
                        client = AnthropicVertex(
                            region=os.environ["VERTEX_LOCATION"],
                            project_id=os.environ["VERTEX_PROJECT"],
                        )
                        message = client.messages.create(
                            model=self.model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=system_message,
                            messages=[message[1]],
                        )
                        message_json_str = message.model_dump_json(indent=2)
                        message_dict = json.loads(message_json_str)
                        text_content = message_dict["content"][0]["text"]
                        if verbose:
                            print(
                                "Prompt usage:",
                                message_dict["usage"]["input_tokens"],
                                f"${message_dict['usage']['input_tokens']/1000000*3}",
                            )
                            print(
                                "Prompt usage:",
                                message_dict["usage"]["output_tokens"],
                                f"${message_dict['usage']['output_tokens']/1000000*15}",
                            )
                        return text_content
                    else:
                        config = GenerationConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        # safety config
                        safety_config = [
                            SafetySetting(
                                category=VertexHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=VertexHarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=VertexHarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=VertexHarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=VertexHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=VertexHarmBlockThreshold.BLOCK_NONE,
                            ),
                            SafetySetting(
                                category=VertexHarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=VertexHarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]

                        response = self.model_obj.generate_content(
                            system_message
                            + prompt,  # Didn't find a way to add system message in the API
                            generation_config=config,
                            safety_settings=safety_config,
                        )
                        return response.text.strip()


                elif self.api == "vllm":
                    from vllm import SamplingParams
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop_token_ids=[
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                        ],
                    )
                    final_prompt = self.tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    vllm_output = self.llm.generate([final_prompt], sampling_params)
                    return [output.outputs[0].text for output in vllm_output][0]
                
                elif self.api == "claude_code":
                    # Vendored addition (SPEC §5.6.2): Opus 4.7 via
                    # Claude Code Max subscription. The CLI doesn't expose
                    # temperature / top_p / max_tokens, so those args are
                    # accepted-and-ignored to keep the call-site signature
                    # uniform. Usage is estimated via tiktoken since the CLI
                    # doesn't return token counts.
                    from benchmarking.llm_clients.claude_code import call_claude

                    full_prompt = f"{system_message}\n\n{prompt}"
                    response_text = call_claude(
                        full_prompt,
                        model=self.model,
                        log_prefix=f"[topicgpt/{self.api}]",
                    )
                    self.usage["n_calls"] += 1
                    self.usage["input_tokens"] += self.estimate_token_count(full_prompt)
                    self.usage["output_tokens"] += self.estimate_token_count(response_text)
                    return response_text

                elif self.api == "anthropic":
                    # Vendored addition (SPEC §5.6.2): metered sync Anthropic
                    # SDK. Used for correction (small N). Bulk per-doc
                    # assignment goes through the Batch API, not here ---
                    # see benchmarking/baselines/topicgpt/batch_assigner.py.
                    msg = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        system=system_message,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    self.usage["n_calls"] += 1
                    self.usage["input_tokens"] += msg.usage.input_tokens
                    self.usage["output_tokens"] += msg.usage.output_tokens
                    cache_read = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
                    cache_creation = getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
                    self.usage["cache_read_input_tokens"] += cache_read
                    self.usage["cache_creation_input_tokens"] += cache_creation
                    if verbose:
                        print(
                            f"[anthropic] in={msg.usage.input_tokens} "
                            f"out={msg.usage.output_tokens} "
                            f"cache_r={cache_read} cache_w={cache_creation}"
                        )
                    return msg.content[0].text

                elif self.api == "gemini":
                    if not _HAS_GEMINI:
                        raise ImportError("gemini backend requires google-generativeai.")
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                    self.model_obj = genai.GenerativeModel(self.model)
                    config = genai.types.GenerationConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    # safety config
                    safety_config = {
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                          HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                          HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                          HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                          }
                    response = self.model_obj.generate_content(
                        system_message
                        + prompt,  # Didn't find a way to add system message in the API
                        generation_config=config,
                        safety_settings=safety_config,
                    )
                    return response.text.strip()

            except Exception as e:
                print(f"Attempt {attempt + 1}/{num_try} failed: {e}")
                if attempt < num_try - 1:
                    time.sleep(60)  # avoid rate limiting issues
                else:
                    raise

    def batch_prompt(
        self,
        prompts: list,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        system_message: str = "You are a helpful assistant.",
    ):
        """
        Batch prompting for vLLM API

        Parameters:
        - prompts: List of prompts
        - max_tokens: Maximum token count
        - temperature: Temperature for sampling
        - top_p: Top p value for sampling
        - system_message: System message

        Returns:
        - responses: List of response texts
        """
        if self.api != "vllm":
            raise ValueError("Batch prompting not supported for this API.")

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )

        prompt_formatted = [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        final_prompts = [
            self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in prompt_formatted
        ]
        outputs = self.llm.generate(final_prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class TopicTree:
    """
    Represents a hierarchical structure of topics.

    Parameters:
    - root_name: Name of the root topic

    Attributes:
    - root: Root node of the tree
    - level_nodes: Dictionary of nodes by level

    Methods:
    - node_to_str: Convert a node to a string representation
    - from_topic_list: Construct a TopicTree from a list of topic strings
    - from_seed_file: Construct a TopicTree from a seed file
    - _add_node: Add a node to the tree
    - _remove_node_by_name_lvl: Remove a node by name and level
    - to_prompt_view: Generate a string representation of the tree with indentation by level
    - find_duplicates: Find nodes with the same name and level in the tree
    - to_file: Save the tree to a file
    - to_topic_list: Convert the tree to a list of topic strings
    - get_root_descendants_name: Get the root description
    - update_tree: Update the topic tree by merging a set of topics into a new topic
    """

    def __init__(self, root_name="Topics"):
        self.root = Node(name=root_name, lvl=0, count=1, desc="Root topic", parent=None)
        self.level_nodes = {0: self.root}

    @staticmethod
    def node_to_str(node, count=True, desc=True):
        """
        Convert a node to a string representation.

        Parameters:
        - node: Node to convert
        - count: Include count in the string
        - desc: Include description in the string

        Returns:
        - str: String representation of the node
        """
        if not count and not desc:
            return f"[{node.lvl}] {node.name}"
        elif not count and desc:
            return f"[{node.lvl}] {node.name}: {node.desc}"
        elif count and not desc:
            return f"[{node.lvl}] {node.name} (Count: {node.count})"
        else:
            return f"[{node.lvl}] {node.name} (Count: {node.count}): {node.desc}"

    @staticmethod
    def from_topic_list(topic_src, from_file=False):
        """
        Construct a TopicTree from a list of topic strings or a file.

        Parameters:
        - topic_src: List of topic strings or path to a file
        - from_file: Flag to indicate if the source is a file

        Returns:
        - tree: Constructed TopicTree
        """
        tree = TopicTree()
        if from_file:
            with open(topic_src, "r") as f:
                topic_list = f.readlines()
        else:
            topic_list = topic_src
        topic_list = [topic for topic in topic_list if len(topic.strip()) > 0]
        pattern = regex.compile(r"^\[(\d+)\] (.+) \(Count: (\d+)\)\s?:(.+)?")

        for topic in topic_list:
            if not topic.strip():
                continue
            try:
                match = regex.match(pattern, topic.strip())
                lvl, label, count, desc = (
                    int(match.group(1)),
                    match.group(2).strip(),
                    int(match.group(3)),
                    match.group(4).strip() if match.group(4) else "",
                )
            except Exception:
                print("Error reading", topic)
                traceback.print_exc()
                continue

            tree._add_node(lvl, label, count, desc, tree.level_nodes.get(lvl - 1))

        return tree

    def from_seed_file(self, seed_file):
        """
        Construct a TopicTree from a seed file (no description/count)

        Parameters:
        - seed_file: Path to the seed file

        Returns:
        - tree: Constructed TopicTree
        """
        tree = TopicTree()
        if seed_file:
            with open(seed_file, "r") as f:
                topic_list = f.readlines()
        else:
            topic_list = []
        topic_list = [topic for topic in topic_list if len(topic.strip()) > 0]
        pattern = regex.compile(r"^\[(\d+)\] (.+)")

        for topic in topic_list:
            if not topic.strip():
                continue
            try:
                match = regex.match(pattern, topic.strip())
                lvl, label = (
                    int(match.group(1)),
                    match.group(2).strip(),
                )
            except Exception:
                print("Error reading", topic)
                traceback.print_exc()
                continue

            tree._add_node(lvl, label, 1, "", tree.level_nodes.get(lvl - 1))

        return tree

    def _add_node(self, lvl, label, count, desc, parent_node):
        """
        Add a node to the tree, merging with duplicates if present.

        Parameters:
        - lvl: Level of the node
        - label: Name of the node
        - count: Count of the node
        - desc: Description of the node
        - parent_node: Parent node of the new node
        """
        if parent_node:
            existing = next((n for n in parent_node.children if n.name == label), None)
            if existing:
                existing.count += count
            else:
                new_node = Node(
                    name=label, lvl=lvl, count=count, desc=desc, parent=parent_node
                )
                self.level_nodes[lvl] = new_node

    def _remove_node_by_name_lvl(self, name, lvl):
        """
        Remove a node by name and level.

        Parameters:
        - name: Name of the node
        - lvl: Level of the node
        """
        node = next(
            (n for n in self.root.descendants if n.name == name and n.lvl == lvl), None
        )
        if node:
            node.parent = None

    def to_prompt_view(self, desc=True):
        """
        Generate a string representation of the tree with indentation by level.

        Parameters:
        - desc: Include description in the string

        Returns:
        - str: String representation of the tree
        """

        def traverse(node, result=""):
            if node.lvl > 0:
                result += (
                    "\t" * (node.lvl - 1)
                    + self.node_to_str(node, count=False, desc=False)
                    + "\n"
                )
            for child in node.children:
                result = traverse(child, result)
            return result

        return traverse(self.root)

    def find_duplicates(self, name, level):
        """
        Find nodes with the same name and level in the tree.

        Parameters:
        - name: Name of the node
        - level: Level of the node

        Returns:
        - list: List of nodes with the same name and level
        """
        return [
            node
            for node in self.root.descendants
            if node.name.lower() == name.lower() and node.lvl == level
        ]

    def to_file(self, fname):
        """
        Save the tree to a file.

        Parameters:
        - fname: Path to the file
        """
        # Vendored edit: force utf-8 (upstream uses the platform default,
        # which is cp1252 on Windows and crashes on common LLM-output chars
        # like ‑ non-breaking hyphen). See CHANGES.md.
        with open(fname, "w", encoding="utf-8") as f:
            for node in self.root.descendants:
                if len(node.desc) > 0:
                    indentation = "    " * (node.lvl - 1)
                    f.write(indentation + self.node_to_str(node) + "\n")

    def to_topic_list(self, desc=True, count=True):
        """
        Convert the tree to a list of topic strings.

        Parameters:
        - desc: Include description in the string
        - count: Include count in the string

        Returns:
        - list: List of topic strings
        """
        return [self.node_to_str(node, count, desc) for node in self.root.descendants]

    def get_root_descendants_name(self):
        """Get the root description.

        Returns:
        - list: List of root descendants' names
        """
        return [node.name for node in self.root.descendants]

    def update_tree(self, original_topics, new_topic_name, new_topic_desc):
        """
        Update the topic tree by merging a set of topics into a new topic if
        the final count is <= the total count of all topics.

        Parameters:
        - original_topics: List of tuples (name, level) of topics to merge.
        - new_topic_name: Name of the new merged topic.
        - new_topic_desc: Description for the new merged topic.

        Returns:
        - tree: Updated TopicTree
        """
        total_count = 0
        parent_node = None
        nodes_to_merge = []

        for name, lvl in original_topics:
            duplicates = self.find_duplicates(name, lvl)
            nodes_to_merge.extend(duplicates)
            total_count += sum(node.count for node in duplicates)
            if duplicates and not parent_node:
                parent_node = duplicates[0].parent

        if parent_node is None:
            parent_node = self.root

        merged_topic_node = (
            next(
                (node for node in parent_node.children if node.name == new_topic_name),
                None,
            )
            if parent_node
            else None
        )

        if merged_topic_node:
            merged_topic_node.count = total_count
            merged_topic_node.desc = new_topic_desc
        else:
            merged_topic_node = self._add_node(
                lvl=parent_node.lvl + 1,
                label=new_topic_name,
                count=total_count,
                desc=new_topic_desc,
                parent_node=parent_node,
            )

        for node in nodes_to_merge:
            if node != merged_topic_node:
                self._remove_node_by_name_lvl(node.name, node.lvl)

        return self


def calculate_purity(true_col, pred_col, df):
    """
    Calculate harmonic purity between two set of clusterings

    Parameters:
    - true_col: Column containing a ground-truth label for each document
    - pred_col: Column containing a predicted label for each document (containing parsed topics)
    - df: Pandas data frame containing two columns (true_col and pred_col)

    Returns:
    - purity: Purity score
    - inverse_purity: Inverse purity score
    - harmonic_purity: Harmonic purity score
    """
    contingency_matrix = metrics.cluster.contingency_matrix(df[true_col], df[pred_col])
    precision = contingency_matrix / contingency_matrix.sum(axis=0).reshape(1, -1)
    recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    purity = (
        np.amax(precision, axis=0) * contingency_matrix.sum(axis=0)
    ).sum() / contingency_matrix.sum()
    inverse_purity = (
        np.amax(recall, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    harmonic_purity = (
        np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    return (purity, inverse_purity, harmonic_purity)


def calculate_metrics(true_col, pred_col, df):
    """
    Calculate topic alignment between df1 and df2 (harmonic purity, ARI, NMI)

    Parameters:
    - true_col: Column containing a ground-truth label for each document
    - pred_col: Column containing a predicted label for each document (containing parsed topics)
    - df: Pandas data frame containing two columns (true_col and pred_col)

    Returns:
    - harmonic_purity: Harmonic purity score
    - ari: Adjusted Rand Index
    - mis: Normalized Mutual Information
    """
    _, _, harmonic_purity = calculate_purity(true_col, pred_col, df)
    ari = metrics.adjusted_rand_score(df[true_col], df[pred_col])
    mis = metrics.normalized_mutual_info_score(df[true_col], df[pred_col])
    return (harmonic_purity, ari, mis)
