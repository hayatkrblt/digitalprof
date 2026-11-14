"""ActiveSlide - Streamlit frontend."""
import base64
import json
import time

import streamlit as st

from src.agent import get_client, process_document
from src.input import InputError, MAX_PAGES, load_pages
from src.output import build_response, render_pdf

st.set_page_config(page_title="ActiveSlide", page_icon="📑", layout="wide")
st.title("📑 ActiveSlide")
st.caption(
    "Upload lecture slides and get a varied version that breaks visual memorization. "
    "Core terms and diagrams stay exactly as they are."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Input** – validate and rasterize\n2. **Knowledge** – preservation rules\n"
        "3. **Tools** – categorize and mask\n4. **Decisions** – route or refuse\n5. **Output** – JSON and PDF"
    )
    st.markdown(f"Limit: {MAX_PAGES} pages per upload.")

upload = st.file_uploader("Slides (PDF, PNG, JPG, WEBP)", type=["pdf", "png", "jpg", "jpeg", "webp"])
query = st.text_input("Optional instruction", placeholder="e.g. Make it a quiz-focused version")

if upload is not None:
    try:
        pages = load_pages(upload.name, upload.getvalue())
    except InputError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader(f"Original ({len(pages)} page{'s' if len(pages) != 1 else ''})")
    if upload.name.lower().endswith(".pdf"):
        encoded = base64.b64encode(upload.getvalue()).decode("ascii")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="520"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.image(pages[0].png_bytes, use_container_width=True)

    if st.button("Generate active-recall version", type="primary"):
        try:
            client, model = get_client()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        progress = st.progress(0.0, text="Starting...")
        start = time.perf_counter()
        try:
            results = process_document(
                client, model, pages, query,
                on_page=lambda done, total: progress.progress(done / total, text=f"Processed {done}/{total}"),
            )
            pdf = render_pdf(results, pages)
        except Exception as exc:
            progress.empty()
            st.error(f"Processing failed: {exc}")
            st.stop()
        progress.empty()
        st.session_state["results"] = results
        st.session_state["response"] = build_response(results, time.perf_counter() - start)
        st.session_state["pdf"] = pdf

response = st.session_state.get("response")
if response:
    st.divider()
    st.subheader("Result")
    pdf = st.session_state.get("pdf")
    if pdf:
        st.download_button("⬇️ Download PDF", pdf, "activeslide.pdf", "application/pdf", type="primary")
        encoded = base64.b64encode(pdf).decode("ascii")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="560"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Nothing to render: every page was refused.")

    for page in response["pages"]:
        with st.expander(f"Page {page['page_number']} · {page['action']}", expanded=page["action"] == "refuse"):
            if page["action"] == "refuse":
                st.warning(page["refusal_message"])
            else:
                st.markdown(f"**{page['title']}**")
                for paragraph in page["paragraphs"]:
                    st.write(paragraph)
            st.caption(page["reason"])
            for warning in page["warnings"]:
                st.info(warning)
            for index, question in enumerate(page["masks"]):
                st.markdown(f"**Q{index + 1}.** {question['sentence'].replace('[MASK]', '_____')}")
                key = f"q_{page['page_number']}_{index}"
                choice = st.radio("Pick one", question["options"], index=None, key=key, label_visibility="collapsed")
                if choice is not None:
                    (st.success if choice == question["answer"] else st.error)(
                        "Correct!" if choice == question["answer"] else f"Not quite - answer: {question['answer']}"
                    )

    with st.expander("JSON response and debug trace"):
        st.code(json.dumps(response, indent=2), language="json")
