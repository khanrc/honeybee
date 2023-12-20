# Reference: https://huggingface.co/spaces/MAGAer13/mPLUG-Owl/tree/main
import argparse
import json
import os
import time

import gradio as gr
import requests
import torch

from .conversation import default_conversation
from .gradio_css import code_highlight_css
from .model_utils import post_process_code
from .model_worker import Honeybee_Server

from .serve_utils import add_text  # noqa: F401
from .serve_utils import regenerate  # noqa: F401
from .serve_utils import (
    after_process_image,
    clear_history,
    disable_btn,
    downvote_last_response,
    enable_btn,
    flag_last_response,
    get_window_url_params,
    init,
    no_change_btn,
    upvote_last_response,
)


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line.strip("\n")) for line in f.readlines()]


def set_dataset(example: list):
    return gr.Image.update(value=example[0]), gr.Textbox.update(value=example[1])


def set_example_text_input(example_text: str) -> dict:
    # for the example query texts
    return gr.Textbox.update(value=example_text[0])


def load_demo(url_params, request: gr.Request):
    dropdown_update = gr.Dropdown.update(visible=True)
    state = default_conversation.copy()

    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def add_text_http_bot(
    state,
    text,
    image,
    video,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
):
    if len(text) <= 0 and (image is None or video is None):
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5

    if image is not None:
        if "<image>" not in text:
            text = text + "\n<image>"
        text = (text, image)

    if video is not None:
        num_frames = 4
        if "<image>" not in text:
            text = text + "\n<image>" * num_frames
        text = (text, video)

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False

    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5
        return

    prompt = after_process_image(state.get_prompt())
    images = state.get_images()

    data = {
        "text_input": prompt,
        "images": images if len(images) > 0 else [],
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        },
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    try:
        for chunk in model.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:  # noqa: F841
        state.messages[-1][
            -1
        ] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5


def regenerate_http_bot(
    state,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
):
    state.messages[-1][-1] = None
    state.skip_next = False
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    prompt = after_process_image(state.get_prompt())
    images = state.get_images()

    data = {
        "text_input": prompt,
        "images": images if len(images) > 0 else [],
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        },
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    try:
        for chunk in model.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:  # noqa: F841
        state.messages[-1][
            -1
        ] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5


title_markdown = """
**Notice**: The output is generated by top-k sampling scheme and may involve some randomness.
"""

tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

**Copyright 2023 Alibaba DAMO Academy.**
"""

learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""

css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""
)


def build_demo(model_name: str = "M-LLM"):
    title_model_name = f"""<h1 align="center">{model_name} </h1>"""
    # with gr.Blocks(title="mPLUG-Owl🦉", theme=gr.themes.Base(), css=css) as demo:
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", visible=False, container=False
    )
    with gr.Blocks(title="M-LLM", css=css) as demo:
        state = gr.State()

        gr.Markdown(title_model_name)
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Image(type="pil")

                # dataset for selecting OwlEval data
                owleval = load_jsonl("data/OwlEval/questions.jsonl")
                owleval_data = gr.Dataset(
                    components=[imagebox, textbox],
                    label="OwlEval Examples",
                    samples=[
                        [os.path.join("data/OwlEval", "images", it["image"]), it["question"]]
                        for it in owleval
                    ],
                )

                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    max_output_tokens = gr.Slider(
                        0, 1024, 512, step=64, interactive=True, label="Max output tokens"
                    )
                    temperature = gr.Slider(
                        0, 1, 1, step=0.1, interactive=True, label="Temperature"
                    )
                    top_k = gr.Slider(1, 5, 3, step=1, interactive=True, label="Top K")
                    top_p = gr.Slider(0, 1, 0.9, step=0.1, interactive=True, label="Top p")
                    length_penalty = gr.Slider(
                        1, 5, 1, step=0.1, interactive=True, label="length_penalty"
                    )
                    num_beams = gr.Slider(1, 5, 1, step=1, interactive=True, label="Beam Size")
                    no_repeat_ngram_size = gr.Slider(
                        1, 5, 2, step=1, interactive=True, label="no_repeat_ngram_size"
                    )
                    do_sample = gr.Checkbox(interactive=True, value=True, label="do_sample")

                videobox = gr.Video(visible=False)  # [M-LLM] currently, we do not support video

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(elem_id="chatbot", visible=False, height=1000)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        owleval_data.click(fn=set_dataset, inputs=owleval_data, outputs=owleval_data.components)
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        parameter_list = [
            max_output_tokens,
            temperature,
            top_k,
            top_p,
            num_beams,
            no_repeat_ngram_size,
            length_penalty,
            do_sample,
        ]
        upvote_btn.click(
            upvote_last_response, [state], [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        downvote_btn.click(
            downvote_last_response, [state], [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        flag_btn.click(flag_last_response, [state], [textbox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(
            regenerate_http_bot,
            [state] + parameter_list,
            [state, chatbot, textbox, imagebox, videobox] + btn_list,
        )

        clear_btn.click(
            clear_history, None, [state, chatbot, textbox, imagebox, videobox] + btn_list
        )

        textbox.submit(
            add_text_http_bot,
            [state, textbox, imagebox, videobox] + parameter_list,
            [state, chatbot, textbox, imagebox, videobox] + btn_list,
        )

        submit_btn.click(
            add_text_http_bot,
            [state, textbox, imagebox, videobox] + parameter_list,
            [state, chatbot, textbox, imagebox, videobox] + btn_list,
        )

        demo.load(
            load_demo,
            [url_params],
            [state, chatbot, textbox, submit_btn, button_row, parameter_row],
            _js=get_window_url_params,
        )

    return demo


if __name__ == "__main__":
    io = init()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = cur_dir[:-9] + "log"

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency-count", type=int, default=100)
    parser.add_argument("--base-model", type=str, default="checkpoints/7B-C-Abs-M144/last")
    parser.add_argument("--load-8bit", action="store_true", help="using 8bit mode")
    parser.add_argument("--bf16", action="store_true", help="using 8bit mode")
    args = parser.parse_args()

    print(" >>> Init server")
    model = Honeybee_Server(
        base_model=args.base_model,
        log_dir=log_dir,
        load_in_8bit=args.load_8bit,
        bf16=args.bf16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        io=io,
    )
    print(" --- Done init server")

    demo = build_demo(model_name="Honeybee")
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(server_name=args.host, debug=args.debug, server_port=args.port, share=False)
