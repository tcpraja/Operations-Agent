import os
import base64
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

API_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
)

APP_API_KEY = os.getenv(
    "APP_API_KEY",
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Manufacturing Operations AI Agent",
    page_icon="🏭",
    layout="wide",
)


# =========================================================
# VALIDATE CONFIGURATION
# =========================================================

if not APP_API_KEY:
    st.error(
        "APP_API_KEY is not configured."
    )
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    # Each message: {"role": "user"/"assistant",
    #                 "type": "text"/"result",
    #                 "content": str or dict}
    st.session_state.messages = []

# Earlier versions displayed provisional analyses while
# awaiting clarification. Hide those in existing sessions.
st.session_state.messages = [
    message
    for message in st.session_state.messages
    if not (
        message.get("type") == "result"
        and message.get("content", {}).get("needs_clarification")
    )
]


# =========================================================
# CONVERSATION ID (so past conversations survive a refresh
# or a new browser session)
# =========================================================

if "session_id" in st.query_params:
    st.session_state.session_id = st.query_params["session_id"]
elif "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.query_params["session_id"] = st.session_state.session_id


def expand_chat_response(result: dict) -> list[dict]:
    """Turn one /chat response into the one-or-more message
    entries the chat UI renders (text, image, result, pareto,
    web_images, chart)."""

    expanded = []
    if result.get("reply"):
        expanded.append({"role": "assistant", "type": "text", "content": result["reply"]})
    if result.get("image_svg"):
        expanded.append({"role": "assistant", "type": "image", "content": result})
    if result.get("analysis"):
        expanded.append({"role": "assistant", "type": "result", "content": result["analysis"]})
    if result.get("pareto"):
        expanded.append({"role": "assistant", "type": "pareto", "content": result["pareto"]})
    if result.get("web_images"):
        expanded.append({"role": "assistant", "type": "web_images", "content": result["web_images"]})
    if result.get("chart_image"):
        expanded.append({"role": "assistant", "type": "chart", "content": result["chart_image"]})
    if result.get("download"):
        expanded.append({"role": "assistant", "type": "file_download", "content": result["download"]})
    return expanded


def load_conversation_history(session_id: str) -> list[dict]:
    """Fetch a previously stored conversation from the backend
    and rebuild it into the chat UI's message list."""

    try:
        response = requests.get(
            f"{API_URL}/conversations/{session_id}",
            headers={"X-API-Key": APP_API_KEY},
            timeout=15,
        )
        response.raise_for_status()
        stored_messages = response.json().get("messages", [])
    except requests.exceptions.RequestException:
        return []

    rebuilt = []
    for stored in stored_messages:
        role, msg_type, content = stored["role"], stored["type"], stored["content"]
        if msg_type == "chat_response":
            # Legacy rows saved before per-part persistence existed.
            rebuilt.extend(expand_chat_response(content))
        else:
            # Every other stored type (text, image, result, pareto,
            # web_images, chart, file_download, upload) round-trips
            # directly: it was saved in the exact shape the UI renders.
            rebuilt.append({"role": role, "type": msg_type, "content": content})
    return rebuilt


def flatten_history(messages: list[dict]) -> list[dict]:
    """Collapse the UI's rich message list into plain
    {role, content} turns for the /chat API's history field."""

    history = []
    for previous in messages:
        content = previous["content"]
        if previous["type"] == "image":
            content = content["reply"]
        elif previous["type"] == "result":
            content = content.get("summary", "")
        elif previous["type"] == "pareto":
            content = f"Pareto chart from {content['source']}: {content['note']}"
        elif previous["type"] == "upload":
            content = content.get("reply", "")
        elif previous["type"] == "web_images":
            content = "Shared real web photos for: " + ", ".join(
                image_result.get("title", "") for image_result in content
            )
        elif previous["type"] == "chart":
            content = "Shared a chart."
        elif previous["type"] == "file_download":
            content = f"Generated a downloadable file: {content.get('filename', '')}"
        if content:
            history.append({"role": previous["role"], "content": content})
    return history


def save_conversation_snapshot(messages: list[dict]) -> None:
    """Persist the current full message list as-is, replacing
    whatever was previously stored for this session."""

    try:
        requests.put(
            f"{API_URL}/conversations/{st.session_state.session_id}",
            json={"messages": messages},
            headers={"X-API-Key": APP_API_KEY},
            timeout=15,
        )
    except requests.exceptions.RequestException:
        pass


if "loaded_session_id" not in st.session_state:
    st.session_state.loaded_session_id = None

if "editing_message_index" not in st.session_state:
    st.session_state.editing_message_index = None

if st.session_state.loaded_session_id != st.session_state.session_id:
    st.session_state.messages = load_conversation_history(st.session_state.session_id)
    st.session_state.loaded_session_id = st.session_state.session_id


# =========================================================
# HEADER (frozen at the top while the chat scrolls)
# =========================================================

st.markdown(
    """
    <style>
    div.st-key-sticky_header {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #002E61;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 1rem;
    }
    div.st-key-sticky_header h1,
    div.st-key-sticky_header p {
        color: #FFFFFF;
    }
    /* Right-align the user's own messages (avatar + bubble),
       like a normal chat app; assistant replies stay left. */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        justify-content: flex-end;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
    div[data-testid="stChatMessageContent"] * {
        text-align: right !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="sticky_header"):

    st.title(
        "🏭 Manufacturing Operations AI Agent"
    )

    st.caption(
        (
            "Ask about manufacturing equipment, incidents, "
            "or request a simple equipment diagram."
        )
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    if st.button(
        "New",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.loaded_session_id = st.session_state.session_id
        st.query_params["session_id"] = st.session_state.session_id

        st.rerun()

    st.divider()

    st.subheader("Chats")

    try:
        history_response = requests.get(
            f"{API_URL}/conversations",
            headers={"X-API-Key": APP_API_KEY},
            timeout=10,
        )
        history_response.raise_for_status()
        past_conversations = history_response.json().get("conversations", [])
    except requests.exceptions.RequestException:
        past_conversations = []

    if not past_conversations:
        st.caption("No chats yet.")

    if "renaming_session_id" not in st.session_state:
        st.session_state.renaming_session_id = None

    for past in past_conversations[:10]:
        is_current = past["session_id"] == st.session_state.session_id
        label = past.get("title") or past["preview"] or "(no preview)"

        if st.session_state.renaming_session_id == past["session_id"]:
            new_title = st.text_input(
                "Rename chat",
                value=label,
                key=f"rename_input_{past['session_id']}",
                label_visibility="collapsed",
            )
            save_column, cancel_column = st.columns(2)
            with save_column:
                if st.button("Save", key=f"save_rename_{past['session_id']}", use_container_width=True):
                    if new_title.strip():
                        try:
                            requests.put(
                                f"{API_URL}/conversations/{past['session_id']}/title",
                                json={"title": new_title.strip()},
                                headers={"X-API-Key": APP_API_KEY},
                                timeout=10,
                            )
                        except requests.exceptions.RequestException:
                            pass
                    st.session_state.renaming_session_id = None
                    st.rerun()
            with cancel_column:
                if st.button("Cancel", key=f"cancel_rename_{past['session_id']}", use_container_width=True):
                    st.session_state.renaming_session_id = None
                    st.rerun()
            continue

        row_columns = st.columns([4, 1, 1])
        with row_columns[0]:
            if st.button(
                ("• " if is_current else "") + label,
                key=f"conversation_{past['session_id']}",
                use_container_width=True,
                disabled=is_current,
            ):
                st.session_state.session_id = past["session_id"]
                st.session_state.loaded_session_id = None
                st.query_params["session_id"] = past["session_id"]
                st.rerun()
        with row_columns[1]:
            if st.button(
                "✏️",
                key=f"rename_{past['session_id']}",
                use_container_width=True,
            ):
                st.session_state.renaming_session_id = past["session_id"]
                st.rerun()
        with row_columns[2]:
            if st.button(
                "🗑",
                key=f"delete_{past['session_id']}",
                use_container_width=True,
            ):
                try:
                    requests.delete(
                        f"{API_URL}/conversations/{past['session_id']}",
                        headers={"X-API-Key": APP_API_KEY},
                        timeout=10,
                    )
                except requests.exceptions.RequestException:
                    pass
                if past["session_id"] == st.session_state.session_id:
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.loaded_session_id = st.session_state.session_id
                    st.query_params["session_id"] = st.session_state.session_id
                st.rerun()

    if past_conversations:
        st.divider()
        confirm_delete_all = st.checkbox(
            "Confirm delete all chats",
            key="confirm_delete_all_chats",
        )
        if st.button(
            "Delete all chats",
            use_container_width=True,
            disabled=not confirm_delete_all,
        ):
            try:
                requests.delete(
                    f"{API_URL}/conversations",
                    params={"delete_all": "true"},
                    headers={"X-API-Key": APP_API_KEY},
                    timeout=10,
                )
            except requests.exceptions.RequestException:
                pass
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.loaded_session_id = st.session_state.session_id
            st.query_params["session_id"] = st.session_state.session_id
            st.rerun()


# =========================================================
# CHAT API HELPER
# =========================================================

def call_chat_api(message: str, history: list[dict], persist: bool = True):
    """Send a conversational request to the backend.

    persist=False skips the backend's own auto-save — used when the
    caller (e.g. an edited/regenerated message) will persist the
    full corrected conversation itself via save_conversation_snapshot,
    so the turn isn't saved twice."""
    st.session_state.last_api_error = None
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "history": history[-8:],
                "session_id": st.session_state.session_id if persist else None,
            },
            headers={"X-API-Key": APP_API_KEY},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        st.session_state.last_api_error = (
            "I couldn't complete that request. Please try again."
        )
        return None


# =========================================================
# RESULT RENDERING
# =========================================================

def render_analysis_result(
    result: dict,
    message_key: str = "current",
):

    # -----------------------------------------------------
    # KPI CARDS
    # (only shown when the value is actually meaningful)
    # -----------------------------------------------------

    severity = result.get(
        "severity",
        "N/A",
    )

    escalation = result.get(
        "escalation_required",
        False,
    )

    confidence = result.get(
        "confidence"
    )

    estimated_loss = result.get(
        "estimated_loss_usd"
    )

    kpi_metrics = [
        (
            "Severity",
            severity,
        ),
        (
            "Escalation",
            (
                "YES"
                if escalation
                else "NO"
            ),
        ),
    ]

    if confidence is not None:

        kpi_metrics.append(
            (
                "Confidence",
                f"{confidence * 100:.0f}%",
            )
        )

    if estimated_loss:

        kpi_metrics.append(
            (
                "Estimated Loss",
                f"${estimated_loss:,.2f}",
            )
        )

    kpi_columns = st.columns(
        len(kpi_metrics)
    )

    for column, (label, value) in zip(
        kpi_columns,
        kpi_metrics,
    ):

        with column:

            st.metric(
                label,
                value,
            )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary_text = result.get(
        "summary",
        "",
    ).strip()

    if summary_text:

        st.markdown(
            f"**Summary:** {summary_text}"
        )

    # -----------------------------------------------------
    # LIKELY CAUSES
    # -----------------------------------------------------

    likely_causes = result.get(
        "likely_causes",
        [],
    )

    if likely_causes:

        st.markdown(
            "**Likely Causes**"
        )

        for item in likely_causes:

            st.markdown(
                f"- {item}"
            )

    # -----------------------------------------------------
    # IMMEDIATE ACTIONS
    # -----------------------------------------------------

    immediate_actions = result.get(
        "immediate_actions",
        [],
    )

    if immediate_actions:

        st.markdown(
            "**Immediate Actions**"
        )

        for index, item in enumerate(
            immediate_actions,
            start=1,
        ):

            st.markdown(
                f"{index}. {item}"
            )

    # -----------------------------------------------------
    # VERIFICATION CHECKS
    # -----------------------------------------------------

    verification_checks = result.get(
        "verification_checks",
        [],
    )

    if verification_checks:

        st.markdown(
            "**Verification Checks**"
        )

        for item in verification_checks:

            st.markdown(
                f"- {item}"
            )

    # -----------------------------------------------------
    # EQUIPMENT SPECIFICATIONS
    # (only shown when specs were actually identified)
    # -----------------------------------------------------

    equipment_specifications = result.get(
        "equipment_specifications",
        "",
    ).strip()

    if (
        equipment_specifications
        and "not enough information"
        not in equipment_specifications.lower()
    ):

        st.markdown(
            "**Equipment Specifications**"
        )

        st.write(
            equipment_specifications
        )

    # -----------------------------------------------------
    # SUPPLIER AVAILABILITY
    # (only shown when suppliers were actually found)
    # -----------------------------------------------------

    supplier_availability = result.get(
        "supplier_availability",
        [],
    )

    if supplier_availability:

        st.markdown(
            "**Supplier Availability**"
        )

        for item in supplier_availability:

            st.markdown(
                f"- {item}"
            )

    # -----------------------------------------------------
    # SPARE PARTS / COST ESTIMATE
    # (only shown when cost info was actually found)
    # -----------------------------------------------------

    cost_estimate = result.get(
        "spare_parts_cost_estimate",
        "",
    ).strip()

    if (
        cost_estimate
        and "no cost information"
        not in cost_estimate.lower()
    ):

        st.markdown(
            "**Spare Parts / Cost Estimate**"
        )

        st.write(
            cost_estimate
        )

    web_sources = result.get("web_sources", [])

    if web_sources:
        st.markdown("**Web References**")
        st.caption(
            "General equipment references; verify site-specific "
            "causes through inspection."
        )
        for source in web_sources[:5]:
            if source.get("url"):
                st.markdown(
                    f"- [{source.get('title') or source['url']}]"
                    f"({source['url']})"
                )

    # -----------------------------------------------------
    # 80/20 PARETO ANALYSIS OF FAILURE CAUSES
    # -----------------------------------------------------

    pareto_analysis = result.get(
        "pareto_analysis"
    )

    if pareto_analysis:

        st.markdown(
            "**80/20 Pareto Analysis "
            "— Failure Causes**"
        )

        st.caption(
            pareto_analysis.get(
                "summary",
                "",
            )
        )

        table_rows = [
            {
                "Rank": item["rank"],
                "Cause": item["cause"],
                "Count": item["count"],
                "% of Total": item[
                    "percentage"
                ],
                "Cumulative %": item[
                    "cumulative_percentage"
                ],
                "Vital Few (80%)": (
                    "Yes"
                    if item["is_vital_few"]
                    else "No"
                ),
            }
            for item in pareto_analysis.get(
                "items",
                [],
            )
        ]

        if table_rows:

            st.table(
                table_rows
            )

    # -----------------------------------------------------
    # DOWNLOADABLE REPORT
    # -----------------------------------------------------

    st.markdown("**Download report**")
    format_columns = st.columns(4)
    format_labels = {
        "docx": "Word (.docx)",
        "pptx": "PowerPoint (.pptx)",
        "xlsx": "Excel (.xlsx)",
        "pdf": "PDF",
    }
    if "export_cache" not in st.session_state:
        st.session_state.export_cache = {}

    for column, (fmt, label) in zip(format_columns, format_labels.items()):
        with column:
            cache_key = f"{message_key}_{fmt}"
            if cache_key not in st.session_state.export_cache:
                st.session_state.export_cache[cache_key] = call_export_api(result, fmt)
            report = st.session_state.export_cache[cache_key]
            if report:
                st.download_button(
                    label,
                    data=base64.b64decode(report["data_base64"]),
                    file_name=report["filename"],
                    mime=report["mime_type"],
                    key=f"export_btn_{cache_key}",
                    use_container_width=True,
                )


# =========================================================
# FILE UPLOAD API HELPER
# =========================================================

def call_export_api(analysis: dict, fmt: str):
    try:
        response = requests.post(
            f"{API_URL}/export",
            json={"format": fmt, "analysis": analysis},
            headers={"X-API-Key": APP_API_KEY},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def call_upload_api(uploaded_file, instruction: str):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"session_id": st.session_state.session_id}
        if instruction:
            data["instruction"] = instruction
        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            data=data,
            headers={"X-API-Key": APP_API_KEY},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


# =========================================================
# CHAT HISTORY
# =========================================================

for message_index, message in enumerate(st.session_state.messages):

    with st.chat_message(
        message["role"]
    ):

        if message["type"] == "text":

            if (
                message["role"] == "user"
                and st.session_state.editing_message_index == message_index
            ):
                edited_text = st.text_area(
                    "Edit message",
                    value=message["content"],
                    key=f"edit_input_{message_index}",
                    label_visibility="collapsed",
                )
                resend_column, cancel_column = st.columns(2)
                with resend_column:
                    resend_clicked = st.button(
                        "Resend", key=f"resend_{message_index}", use_container_width=True,
                    )
                with cancel_column:
                    if st.button(
                        "Cancel", key=f"cancel_edit_{message_index}", use_container_width=True,
                    ):
                        st.session_state.editing_message_index = None
                        st.rerun()
                if resend_clicked and edited_text.strip():
                    truncated_messages = st.session_state.messages[:message_index]
                    history_for_edit = flatten_history(truncated_messages)
                    with st.spinner("Regenerating..."):
                        result = call_chat_api(edited_text.strip(), history_for_edit, persist=False)
                    new_messages = truncated_messages + [
                        {"role": "user", "type": "text", "content": edited_text.strip()}
                    ]
                    if result is None:
                        new_messages.append({
                            "role": "assistant",
                            "type": "text",
                            "content": st.session_state.last_api_error
                            or "I couldn't complete that request. Please try again.",
                        })
                    else:
                        new_messages.extend(expand_chat_response(result))
                    st.session_state.messages = new_messages
                    st.session_state.editing_message_index = None
                    save_conversation_snapshot(st.session_state.messages)
                    st.rerun()
            else:
                st.markdown(message["content"])
                if message["role"] == "user":
                    if st.button("✏️ Edit", key=f"edit_btn_{message_index}"):
                        st.session_state.editing_message_index = message_index
                        st.rerun()

        elif message["type"] == "image":

            st.markdown(
                message["content"]["image_svg"],
                unsafe_allow_html=True,
            )

        elif message["type"] == "pareto":

            pareto = message["content"]
            st.image(base64.b64decode(pareto["image"].split(",", 1)[1]))
            st.dataframe(
                [{"Failure category": item["category"],
                  "Sites reporting": item["count"],
                  "Cumulative %": item["cumulative_percentage"]}
                 for item in pareto["items"]],
                hide_index=True,
            )
            st.caption(f"Source: {pareto['source']} · {pareto['sheet']} sheet. {pareto['note']}")

        elif message["type"] == "chart":

            st.image(base64.b64decode(message["content"].split(",", 1)[1]))

        elif message["type"] == "file_download":

            download = message["content"]
            st.download_button(
                f"Download {download['filename']}",
                data=base64.b64decode(download["data_base64"]),
                file_name=download["filename"],
                mime=download["mime_type"],
                key=f"file_download_{message_index}",
            )

        elif message["type"] == "web_images":

            st.caption(
                "Real photos found on the web (third-party sources — "
                "verify against your own equipment)."
            )
            image_columns = st.columns(min(len(message["content"]), 4) or 1)
            for column, image_result in zip(image_columns, message["content"]):
                with column:
                    st.image(
                        image_result.get("thumbnail_url") or image_result.get("image_url"),
                        caption=image_result.get("title", ""),
                        use_container_width=True,
                    )
                    if image_result.get("source_url"):
                        st.markdown(f"[Source]({image_result['source_url']})")

        elif message["type"] == "upload":

            upload_result = message["content"]
            st.markdown(upload_result["reply"])
            download = upload_result.get("download")
            if download:
                st.download_button(
                    f"Download {download['filename']}",
                    data=base64.b64decode(download["data_base64"]),
                    file_name=download["filename"],
                    mime=download["mime_type"],
                    key=f"upload_download_{message_index}",
                )

        else:

            render_analysis_result(
                message["content"],
                message_key=str(message_index),
            )


# =========================================================
# CHAT INPUT (text, and optionally an attached PPTX / DOCX /
# XLSX / PDF file to read or edit)
# =========================================================

chat_submission = st.chat_input(
    "Ask me anything about manufacturing, or attach a file to read or edit...",
    accept_file="multiple",
    file_type=["pptx", "docx", "xlsx", "xlsm", "pdf", "txt", "csv", "jpg", "jpeg", "png"],
)

if chat_submission:

    cleaned_message = chat_submission.text.strip()
    attached_files = chat_submission.files

    if attached_files:

        for attached_file in attached_files:

            upload_message = (
                f"Uploaded {attached_file.name}"
                + (f" — {cleaned_message}" if cleaned_message else "")
            )
            st.session_state.messages.append({
                "role": "user",
                "type": "text",
                "content": upload_message,
            })
            with st.chat_message("user"):
                st.markdown(upload_message)

            with st.spinner(f"Processing {attached_file.name}..."):
                upload_result = call_upload_api(attached_file, cleaned_message)

            if upload_result is None:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": "I couldn't process that file. Please try again.",
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "upload",
                    "content": upload_result,
                })

        st.rerun()

if chat_submission and not chat_submission.files and chat_submission.text.strip():

    cleaned_message = chat_submission.text.strip()

    st.session_state.messages.append(
        {
            "role": "user",
            "type": "text",
            "content": cleaned_message,
        }
    )
    with st.chat_message("user"):
        st.markdown(cleaned_message)

    history = flatten_history(st.session_state.messages[:-1])

    with st.spinner("Thinking..."):
        result = call_chat_api(cleaned_message, history)

    if result is None:
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": st.session_state.last_api_error
            or "I couldn't complete that request. Please try again.",
        })
    else:
        st.session_state.messages.extend(expand_chat_response(result))

    st.rerun()
