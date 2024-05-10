import sys
import os
import openai
import hyperdiv as hd

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


def add_message(role, content, state, gpt_model):
    state.messages += (
        dict(role=role, content=content, id=state.message_id, gpt_model=gpt_model),
    )
    state.message_id += 1


def request(gpt_model, state):
    print("run request")
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[dict(role=m["role"], content=m["content"]) for m in state.messages],
        temperature=0,
        stream=True,
    )

    print(f"response: {response}")
    for chunk in response:
        print(f"chunk=        {chunk}")
        message = chunk.choices[0].delta
        print(f"chunk.message={message}")
        state.current_reply += message.content

    add_message("assistant", state.current_reply, state, gpt_model)
    state.current_reply = ""


def render_user_message(content, gpt_model):
    with hd.hbox(
        align="center",
        padding=0.5,
        border_radius="medium",
        background_color="neutral-50",
        font_color="neutral-600",
        justify="space-between",
    ):
        with hd.hbox(gap=0.5, align="center"):
            hd.icon("chevron-right", shrink=0)
            hd.text(content)
        hd.badge(gpt_model)


def main():
    state = hd.state(messages=(), current_reply="", gpt_model="gpt-4", message_id=0)

    task = hd.task()

    template = hd.template(title="GPT Chatbot", sidebar=False)

    with template.body:
        if len(state.messages) > 0:
            with hd.box(direction="vertical-reverse", gap=1.5, vertical_scroll=True):
                if state.current_reply:
                    hd.markdown(state.current_reply)

                for e in reversed(state.messages):
                    with hd.scope(e["id"]):
                        if e["role"] == "system":
                            continue
                        if e["role"] == "user":
                            render_user_message(e["content"], e["gpt_model"])
                        else:
                            hd.markdown(e["content"])

        with hd.box(align="center", gap=1.5):
            # Render the input form.
            with hd.form(direction="horizontal", width="100%") as form:
                with hd.box(grow=1):
                    prompt = form.text_input(
                        placeholder="Talk to the AI",
                        autofocus=True,
                        disabled=task.running,
                        name="prompt",
                    )

                model = form.select(
                    options=("codellama", "llama2", "llama3", "mistral"),
                    value="llama3",
                    name="gpt-model",
                )

            if form.submitted:
                print(f"add message: value={prompt.value} model={model.value}")

                add_message("user", prompt.value, state, model.value)
                prompt.reset()

                print("run request")
                task.rerun(request, model.value, state)

            if len(state.messages) > 0:
                if hd.button(
                    "Start Over", size="small", variant="text", disabled=task.running
                ).clicked:
                    state.messages = ()


hd.run(main)
