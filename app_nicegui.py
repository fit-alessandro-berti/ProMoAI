"""NiceGUI version of the ProMoAI application."""

import os
import shutil
import tempfile

from enum import Enum

from nicegui import ui

import promoai
from promoai.general_utils.ai_providers import (
    AI_MODEL_DEFAULTS,
    AI_HELP_DEFAULTS,
    MAIN_HELP,
    DEFAULT_AI_PROVIDER,
)
from pm4py import (
    read_xes,
    read_pnml,
    read_bpmn,
    convert_to_petri_net,
    convert_to_bpmn,
)
from pm4py.util import constants
from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.objects.bpmn.layout import layouter as bpmn_layouter
from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string


class InputType(Enum):
    TEXT = "Text"
    MODEL = "Model"
    DATA = "Data"


class ViewType(Enum):
    BPMN = "BPMN"
    POWL = "POWL"
    PETRI = "Petri Net"


DISCOVERY_HELP = (
    "The event log will be used to generate a process model using the POWL miner "
    "(see https://doi.org/10.1016/j.is.2024.102493)."
)


def footer() -> None:
    """Add a footer similar to the Streamlit version."""
    style = """
        .footer-container {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 15px 0;
            background-color: white;
            border-top: 2px solid lightgrey;
            z-index: 100;
        }
        .footer-text, .header-text {
            margin: 0;
            padding: 0;
        }
        .footer-links {
            margin: 0;
            padding: 0;
        }
        .footer-links a {
            margin: 0 10px;
            text-decoration: none;
            color: blue;
        }
        .footer-links img {
            vertical-align: middle;
        }
    """

    foot = """
        <div class='footer-container'>
            <div class='footer-text'>
                Developed by
                <a href="https://www.linkedin.com/in/humam-kourani-98b342232/" target="_blank" style="text-decoration:none;">Humam Kourani</a>
                and
                <a href="https://www.linkedin.com/in/alessandro-berti-2a483766/" target="_blank" style="text-decoration:none;">Alessandro Berti</a>
                at the
                <a href="https://www.fit.fraunhofer.de/" target="_blank" style="text-decoration:none;">Fraunhofer Institute for Applied Information Technology FIT</a>.
            </div>
            <div class='footer-links'>
                <a href="https://doi.org/10.24963/ijcai.2024/1014" target="_blank">
                    <img src="https://img.shields.io/badge/ProMoAI:%20Process%20Modeling%20with%20Generative%20AI-gray?logo=googledocs&logoColor=white&labelColor=red" alt="ProMoAI Paper">
                </a>
                <a href="mailto:humam.kourani@fit.fraunhofer.de?cc=a.berti@pads.rwth-aachen.de;" target="_blank">
                    <img src="https://img.shields.io/badge/Email-gray?logo=minutemailer&logoColor=white&labelColor=green" alt="Email Humam Kourani">
                </a>
            </div>
        </div>
    """

    ui.add_head_html(f"<style>{style}</style>")
    ui.add_body_html(foot)


class ProMoAIApp:
    """Main NiceGUI application."""

    def __init__(self) -> None:
        self.temp_dir = "temp"
        self.provider = DEFAULT_AI_PROVIDER
        self.model_name = AI_MODEL_DEFAULTS[self.provider]
        self.model_gen = None
        self.feedback_history: list[str] = []

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        ui.label("🤖 ProMoAI").classes("text-h4")
        ui.label("Process Modeling with Generative AI").classes("text-subtitle1")

        with ui.expansion("Configuration", value=True):
            self.provider_radio = ui.radio(
                [p for p in AI_MODEL_DEFAULTS.keys()],
                value=self.provider,
                on_change=self._update_model_name,
            ).props("inline")
            self.model_input = ui.input(
                "AI model name", value=self.model_name, placeholder="Model name"
            )
            self.api_key_input = ui.input("API key", password=True)

        self.input_type_radio = ui.radio(
            [InputType.TEXT.value, InputType.MODEL.value, InputType.DATA.value],
            value=InputType.TEXT.value,
            on_change=self._update_input_visibility,
        ).props("inline")

        # containers for the various input widgets
        self.text_area = ui.textarea(
            "For process modeling, enter the process description:",
        )

        self.log_upload = ui.upload(
            label="Upload an event log (XES)",
            auto_upload=False,
        ).props("accept=.xes,.gz").style("display:none")

        self.model_upload = ui.upload(
            label="Upload a BPMN or PNML model",
            auto_upload=False,
        ).props("accept=.bpmn,.pnml").style("display:none")

        self.run_button = ui.button("Run", on_click=self._run_generation)

        # outputs
        self.success_message = ui.label().classes("text-green")
        self.feedback_area = ui.textarea("Feedback:")
        self.update_button = ui.button("Update Model", on_click=self._update_model)
        self.feedback_area.set_visibility(False)
        self.update_button.set_visibility(False)

        self.feedback_history_expander = ui.expansion("Feedback History")
        self.feedback_history_expander.set_visibility(False)

        self.download_bpmn_button = ui.button("Download BPMN")
        self.download_pnml_button = ui.button("Download PNML")
        self.download_bpmn_button.set_visibility(False)
        self.download_pnml_button.set_visibility(False)

        self.view_select = ui.select(
            [v.value for v in ViewType],
            value=ViewType.BPMN.value,
            on_change=self._update_view,
        )
        self.view_select.set_visibility(False)
        self.image_container = ui.html("")
        self.image_container.set_visibility(False)

    # ------------------------------------------------------------------
    def _update_model_name(self, e) -> None:
        value = e.value
        self.model_input.value = AI_MODEL_DEFAULTS[value]

    # ------------------------------------------------------------------
    def _update_input_visibility(self, e) -> None:
        value = e.value
        self.text_area.set_visibility(value == InputType.TEXT.value)
        self.log_upload.set_visibility(value == InputType.DATA.value)
        self.model_upload.set_visibility(value == InputType.MODEL.value)

    # ------------------------------------------------------------------
    def _run_generation(self) -> None:
        provider = self.provider_radio.value
        model_name = self.model_input.value
        api_key = self.api_key_input.value
        input_mode = self.input_type_radio.value

        try:
            if input_mode == InputType.TEXT.value:
                description = self.text_area.value
                if not description:
                    ui.notify("Please enter a description", color="negative")
                    return
                process_model = promoai.generate_model_from_text(
                    description,
                    api_key=api_key,
                    ai_model=model_name,
                    ai_provider=provider,
                )
            elif input_mode == InputType.DATA.value:
                if not self.log_upload.value:
                    ui.notify("No event log uploaded", color="negative")
                    return
                uploaded_file = self.log_upload.value[0]
                os.makedirs(self.temp_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, dir=self.temp_dir, suffix=uploaded_file.name
                ) as temp_file:
                    temp_file.write(uploaded_file.content.read())
                    log = read_xes(temp_file.name, variant="rustxes")
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                process_model = promoai.generate_model_from_event_log(log)
            else:
                if not self.model_upload.value:
                    ui.notify("No model uploaded", color="negative")
                    return
                uploaded_file = self.model_upload.value[0]
                ext = uploaded_file.name.split(".")[-1].lower()
                os.makedirs(self.temp_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, dir=self.temp_dir, suffix=uploaded_file.name
                ) as temp_file:
                    temp_file.write(uploaded_file.content.read())
                    path = temp_file.name

                if ext == "bpmn":
                    bpmn_graph = read_bpmn(path)
                    process_model = promoai.generate_model_from_bpmn(bpmn_graph)
                elif ext == "pnml":
                    pn, _, _ = read_pnml(path)
                    process_model = promoai.generate_model_from_petri_net(pn)
                else:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    ui.notify(f"Unsupported file format {ext}", color="negative")
                    return
                shutil.rmtree(self.temp_dir, ignore_errors=True)

            self.model_gen = process_model
            self.success_message.set_text("Model generated successfully!")
            self.feedback_area.set_visibility(True)
            self.update_button.set_visibility(True)
            self.view_select.set_visibility(True)
            self.download_bpmn_button.set_visibility(True)
            self.download_pnml_button.set_visibility(True)
            self.image_container.set_visibility(True)
            self._update_view()
        except Exception as exc:  # pragma: no cover - runtime interaction
            ui.notify(str(exc), color="negative")

    # ------------------------------------------------------------------
    def _update_model(self) -> None:
        if not self.model_gen:
            return
        feedback_text = self.feedback_area.value
        if not feedback_text:
            ui.notify("Please enter feedback", color="warning")
            return
        try:
            self.model_gen.update(
                feedback_text,
                api_key=self.api_key_input.value,
                ai_model=self.model_input.value,
                ai_provider=self.provider_radio.value,
            )
            self.feedback_history.append(feedback_text)
            with self.feedback_history_expander:
                self.feedback_history_expander.clear()
                for i, f in enumerate(self.feedback_history, start=1):
                    ui.label(f"[{i}] {f}")
        except Exception as exc:  # pragma: no cover - runtime interaction
            ui.notify(f"Update failed: {exc}", color="negative")

        self._update_view()

    # ------------------------------------------------------------------
    def _update_view(self, *args) -> None:
        if not self.model_gen:
            return
        try:
            powl = self.model_gen.get_powl()
            pn, im, fm = convert_to_petri_net(powl)
            bpmn = convert_to_bpmn(pn, im, fm)
            bpmn = bpmn_layouter.apply(bpmn)

            view_option = self.view_select.value
            if view_option == ViewType.POWL.value:
                from pm4py.visualization.powl import visualizer

                vis_str = visualizer.apply(powl, parameters={"format": "svg"})
            elif view_option == ViewType.PETRI.value:
                visualization = pn_visualizer.apply(pn, im, fm, parameters={"format": "svg"})
                vis_str = visualization.pipe(format="svg").decode("utf-8")
            else:
                layouted_bpmn = bpmn_layouter.apply(bpmn)
                visualization = bpmn_visualizer.apply(layouted_bpmn, parameters={"format": "svg"})
                vis_str = visualization.pipe(format="svg").decode("utf-8")

            self.image_container.set_content(f"<img src='data:image/svg+xml;utf8,{vis_str}'></img>")

            bpmn_data = get_xml_string(bpmn, parameters={"encoding": constants.DEFAULT_ENCODING})
            pn_data = export_petri_as_string(pn, im, fm)
            self.download_bpmn_button.on("click", lambda: ui.download(bpmn_data, "process_model.bpmn"))
            self.download_pnml_button.on("click", lambda: ui.download(pn_data, "process_model.pnml"))
        except Exception as exc:  # pragma: no cover - runtime interaction
            ui.notify(str(exc), color="negative")


@ui.page("/")
def main_page() -> None:  # pragma: no cover - UI definition
    ProMoAIApp()
    footer()


def run_model_generator_app() -> None:
    """Entry point to run the NiceGUI app."""
    ui.run(title="ProMoAI")


if __name__ == "__main__":  # pragma: no cover - manual run
    run_model_generator_app()
