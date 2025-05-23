import os
import shutil
import tempfile
from pathlib import Path
from nicegui import ui, app
from nicegui.events import UploadEventArguments
import asyncio

import promoai
from promoai.general_utils.app_utils import InputType, ViewType, DISCOVERY_HELP
from promoai.general_utils.ai_providers import AI_MODEL_DEFAULTS, AI_HELP_DEFAULTS, MAIN_HELP, DEFAULT_AI_PROVIDER
from pm4py import read_xes, read_pnml, read_bpmn, convert_to_petri_net, convert_to_bpmn
from pm4py.util import constants
from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.objects.bpmn.layout import layouter as bpmn_layouter
from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string


class ProMoAIApp:
    def __init__(self):
        self.provider = DEFAULT_AI_PROVIDER
        self.model_name = AI_MODEL_DEFAULTS[self.provider]
        self.api_key = ""
        self.selected_mode = InputType.TEXT.value
        self.model_gen = None
        self.feedback_history = []
        self.temp_dir = "temp"
        
        # UI elements references
        self.model_name_input = None
        self.api_key_input = None
        self.description_input = None
        self.feedback_input = None
        self.view_select = None
        self.upload = None
        self.result_container = None
        self.image_container = None
        
        # Check for Graphviz
        self.check_graphviz()
    
    def check_graphviz(self):
        system_dot = shutil.which("dot")
        if system_dot:
            print(f"Found system-wide 'dot' at: {system_dot}")
        else:
            base_path = "/home/adminuser/.conda"
            possible_subpaths = ["bin"]
            
            for sub in possible_subpaths:
                candidate = os.path.join(base_path, sub, "dot")
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    print(f"Found 'dot' at: {candidate}")
                    os.environ["PATH"] += os.pathsep + os.path.dirname(candidate)
                    break
            else:
                ui.notify("Couldn't find 'dot' — is Graphviz installed?", type='warning')
    
    def update_model_name(self):
        self.model_name = AI_MODEL_DEFAULTS[self.provider]
        if self.model_name_input:
            self.model_name_input.value = self.model_name
    
    async def handle_text_generation(self):
        if not self.description_input.value:
            ui.notify("Please enter a process description!", type='negative')
            return
        
        if not self.api_key:
            ui.notify("Please enter an API key!", type='negative')
            return
        
        try:
            ui.notify("Generating model from text...", type='ongoing')
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.model_gen = await loop.run_in_executor(
                None,
                promoai.generate_model_from_text,
                self.description_input.value,
                self.api_key,
                self.model_name,
                self.provider
            )
            
            self.feedback_history = []
            ui.notify("Model generated successfully!", type='positive')
            self.show_results()
            
        except Exception as e:
            ui.notify(f"Error: {str(e)}", type='negative')
    
    async def handle_log_upload(self, e: UploadEventArguments):
        if not e.content:
            ui.notify("No file uploaded!", type='negative')
            return
        
        try:
            ui.notify("Processing event log...", type='ongoing')
            
            # Save uploaded file
            os.makedirs(self.temp_dir, exist_ok=True)
            temp_path = os.path.join(self.temp_dir, e.name)
            
            with open(temp_path, 'wb') as f:
                f.write(e.content.read())
            
            # Read log
            log = read_xes(temp_path, variant="rustxes")
            
            # Generate model
            loop = asyncio.get_event_loop()
            self.model_gen = await loop.run_in_executor(
                None,
                promoai.generate_model_from_event_log,
                log
            )
            
            self.feedback_history = []
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            ui.notify("Model discovered successfully!", type='positive')
            self.show_results()
            
        except Exception as e:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            ui.notify(f"Error during discovery: {str(e)}", type='negative')
    
    async def handle_model_upload(self, e: UploadEventArguments):
        if not e.content:
            ui.notify("No file uploaded!", type='negative')
            return
        
        try:
            ui.notify("Processing model file...", type='ongoing')
            
            file_extension = e.name.split(".")[-1].lower()
            
            # Save uploaded file
            os.makedirs(self.temp_dir, exist_ok=True)
            temp_path = os.path.join(self.temp_dir, e.name)
            
            with open(temp_path, 'wb') as f:
                f.write(e.content.read())
            
            loop = asyncio.get_event_loop()
            
            if file_extension == "bpmn":
                bpmn_graph = read_bpmn(temp_path)
                self.model_gen = await loop.run_in_executor(
                    None,
                    promoai.generate_model_from_bpmn,
                    bpmn_graph
                )
            elif file_extension == "pnml":
                pn, im, fm = read_pnml(temp_path)
                self.model_gen = await loop.run_in_executor(
                    None,
                    promoai.generate_model_from_petri_net,
                    pn
                )
            else:
                raise Exception(f"Unsupported file format: {file_extension}")
            
            self.feedback_history = []
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            ui.notify("Model loaded successfully!", type='positive')
            self.show_results()
            
        except Exception as e:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            ui.notify(f"Error: {str(e)}", type='negative')
    
    async def update_model_with_feedback(self):
        if not self.feedback_input.value:
            ui.notify("Please enter feedback!", type='negative')
            return
        
        if not self.api_key:
            ui.notify("Please enter an API key!", type='negative')
            return
        
        try:
            ui.notify("Updating model...", type='ongoing')
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.model_gen.update,
                self.feedback_input.value,
                self.api_key,
                self.model_name,
                self.provider
            )
            
            self.feedback_history.append(self.feedback_input.value)
            self.feedback_input.value = ""
            
            ui.notify("Model updated successfully!", type='positive')
            self.update_visualization()
            
        except Exception as e:
            ui.notify(f"Update failed: {str(e)}", type='negative')
    
    def download_bpmn(self):
        try:
            powl = self.model_gen.get_powl()
            pn, im, fm = convert_to_petri_net(powl)
            bpmn = convert_to_bpmn(pn, im, fm)
            bpmn = bpmn_layouter.apply(bpmn)
            
            bpmn_data = get_xml_string(bpmn, parameters={"encoding": constants.DEFAULT_ENCODING})
            
            ui.download(bpmn_data, "process_model.bpmn")
        except Exception as e:
            ui.notify(f"Download failed: {str(e)}", type='negative')
    
    def download_pnml(self):
        try:
            powl = self.model_gen.get_powl()
            pn, im, fm = convert_to_petri_net(powl)
            
            pn_data = export_petri_as_string(pn, im, fm)
            
            ui.download(pn_data, "process_model.pnml")
        except Exception as e:
            ui.notify(f"Download failed: {str(e)}", type='negative')
    
    def update_visualization(self):
        if not self.model_gen or not self.view_select:
            return
        
        try:
            view_option = self.view_select.value
            image_format = "svg"
            
            powl = self.model_gen.get_powl()
            pn, im, fm = convert_to_petri_net(powl)
            
            if view_option == ViewType.POWL.value:
                from pm4py.visualization.powl import visualizer
                vis_str = visualizer.apply(powl, parameters={'format': image_format})
                
            elif view_option == ViewType.PETRI.value:
                visualization = pn_visualizer.apply(pn, im, fm, parameters={'format': image_format})
                vis_str = visualization.pipe(format='svg').decode('utf-8')
                
            else:  # BPMN
                bpmn = convert_to_bpmn(pn, im, fm)
                layouted_bpmn = bpmn_layouter.apply(bpmn)
                visualization = bpmn_visualizer.apply(layouted_bpmn, parameters={'format': image_format})
                vis_str = visualization.pipe(format='svg').decode('utf-8')
            
            self.image_container.clear()
            with self.image_container:
                ui.html(vis_str)
                
        except Exception as e:
            ui.notify(f"Visualization error: {str(e)}", type='negative')
    
    def show_results(self):
        self.result_container.clear()
        
        with self.result_container:
            with ui.row().classes('w-full gap-4'):
                # Left column - Feedback
                with ui.column().classes('flex-1'):
                    ui.label('Feedback').classes('text-h6')
                    self.feedback_input = ui.textarea(
                        placeholder='Enter your feedback here...'
                    ).classes('w-full')
                    ui.button('Update Model', on_click=self.update_model_with_feedback).classes('mt-2')
                    
                    if self.feedback_history:
                        with ui.expansion('Feedback History', value=True).classes('mt-4'):
                            for i, feedback in enumerate(self.feedback_history, 1):
                                ui.label(f"[{i}] {feedback}").classes('mb-2')
                
                # Right column - Export and View
                with ui.column().classes('flex-1'):
                    ui.label('Export Model').classes('text-h6')
                    with ui.row():
                        ui.button('Download BPMN', on_click=self.download_bpmn)
                        ui.button('Download PNML', on_click=self.download_pnml)
                    
                    ui.label('Select a view:').classes('mt-4')
                    self.view_select = ui.select(
                        [v_type.value for v_type in ViewType],
                        value=ViewType.POWL.value,
                        on_change=self.update_visualization
                    ).classes('w-full')
            
            # Image visualization
            with ui.expansion('View Image', value=True).classes('w-full mt-4'):
                self.image_container = ui.column().classes('w-full')
                self.update_visualization()
    
    def create_ui(self):
        # Header
        ui.label('🤖 ProMoAI').classes('text-h3')
        ui.label('Process Modeling with Generative AI').classes('text-h5 text-grey-8')
        
        # Configuration
        with ui.expansion('🔧 Configuration', value=True).classes('w-full'):
            ui.label('Choose AI Provider:')
            provider_radio = ui.radio(
                list(AI_MODEL_DEFAULTS.keys()),
                value=self.provider,
                on_change=lambda e: self.on_provider_change(e.value)
            ).props('inline')
            provider_radio.tooltip(MAIN_HELP)
            
            with ui.row().classes('w-full gap-4'):
                with ui.column().classes('flex-1'):
                    self.model_name_input = ui.input(
                        'Enter the AI model name:',
                        value=self.model_name,
                        on_change=lambda e: setattr(self, 'model_name', e.value)
                    ).classes('w-full')
                    self.model_name_input.tooltip(AI_HELP_DEFAULTS[self.provider])
                
                with ui.column().classes('flex-1'):
                    self.api_key_input = ui.input(
                        'API key:',
                        password=True,
                        on_change=lambda e: setattr(self, 'api_key', e.value)
                    ).classes('w-full')
        
        # Input Type Selection
        ui.label('Select Input Type:').classes('mt-4')
        input_type_radio = ui.radio(
            [InputType.TEXT.value, InputType.MODEL.value, InputType.DATA.value],
            value=self.selected_mode,
            on_change=lambda e: self.on_input_type_change(e.value)
        ).props('inline')
        
        # Input forms container
        self.input_container = ui.column().classes('w-full mt-4')
        self.create_input_form()
        
        # Results container
        self.result_container = ui.column().classes('w-full mt-4')
        
        # Footer
        self.create_footer()
    
    def on_provider_change(self, value):
        self.provider = value
        self.update_model_name()
        if self.model_name_input:
            self.model_name_input.tooltip(AI_HELP_DEFAULTS[self.provider])
    
    def on_input_type_change(self, value):
        self.selected_mode = value
        self.model_gen = None
        self.feedback_history = []
        self.result_container.clear()
        self.create_input_form()
    
    def create_input_form(self):
        self.input_container.clear()
        
        with self.input_container:
            if self.selected_mode == InputType.TEXT.value:
                ui.label('For process modeling, enter the process description:')
                self.description_input = ui.textarea().classes('w-full')
                ui.button('Run', on_click=self.handle_text_generation).classes('mt-2')
                
            elif self.selected_mode == InputType.DATA.value:
                ui.label('For process model discovery, upload an event log:')
                self.upload = ui.upload(
                    on_upload=self.handle_log_upload,
                    auto_upload=True
                ).props('accept=.xes,.gz').classes('w-full')
                self.upload.tooltip(DISCOVERY_HELP)
                
            elif self.selected_mode == InputType.MODEL.value:
                ui.label('For process model improvement, upload a semi-block-structured BPMN or Petri net:')
                self.upload = ui.upload(
                    on_upload=self.handle_model_upload,
                    auto_upload=True
                ).props('accept=.bpmn,.pnml').classes('w-full')
    
    def create_footer(self):
        with ui.footer().classes('bg-white').style('border-top: 2px solid lightgrey'):
            with ui.column().classes('w-full items-center'):
                with ui.row():
                    ui.label('Developed by')
                    ui.link('Humam Kourani', 'https://www.linkedin.com/in/humam-kourani-98b342232/')
                    ui.label('and')
                    ui.link('Alessandro Berti', 'https://www.linkedin.com/in/alessandro-berti-2a483766/')
                    ui.label('at the')
                    ui.link('Fraunhofer Institute for Applied Information Technology FIT', 'https://www.fit.fraunhofer.de/')
                
                with ui.row().classes('gap-4 mt-2'):
                    ui.link(
                        'ProMoAI: Process Modeling with Generative AI',
                        'https://doi.org/10.24963/ijcai.2024/1014'
                    ).classes('px-4 py-2 bg-red-500 text-white rounded')
                    
                    ui.link(
                        'Email',
                        'mailto:humam.kourani@fit.fraunhofer.de?cc=a.berti@pads.rwth-aachen.de;'
                    ).classes('px-4 py-2 bg-green-500 text-white rounded')


# Main application
@ui.page('/')
def main():
    app_instance = ProMoAIApp()
    app_instance.create_ui()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="ProMoAI",
        favicon="🤖",
        port=8080
    )