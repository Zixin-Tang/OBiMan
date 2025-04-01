from llm_api import LLM_API
from llm_api import read_prompt_from_file, encode_image
import asyncio
import re
import json
import copy
from PIL import Image
import cv2
import numpy as np
import os
import shutil



Load_Grounded_Info = True


OBiMan_Bench_Root = None
Task_Name = None
Task_GT_Step = None

Grounded_Root = None
Save_Root = None


class TaskSimulator:
    def __init__(self, agent_type, model_name, temperature):
        self.llm = LLM_API(agent_type, model_name, temperature)
        self.objects = dict()  # VLM-based
        self.hands = dict()  # rule-based
        self.objective = dict()  # rule-based
        self.multi_round_dialog_times = 15
    
        self.scenario_description_prompt = read_prompt_from_file("prompts/scenario_description_system.txt")

        self.scenario_description_output_format_prompt = read_prompt_from_file("prompts/scenario_description_output.txt")

        self.messages = None
    
    async def llm_invoke(self, messages):
        res = ''
        async for chunk in self.llm.astream_completion(messages):
            if chunk is not None:
                res += chunk
        return res

    
    def construct_scenario_description_message(self, task_instruction, image):
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant. You must follow the defined output format when answering."},
            {
                "role": "user",
                "content": [
                                {
                                    "type": "text",
                                    "text": self.scenario_description_prompt
                                },
                                {
                                    "type": "text",
                                    "text": f"The queried task instruction is: {task_instruction}"
                                },
                                {
                                    "type": "text",
                                    "text": "The observation image is:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image(image)}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": f"Please output your answer in the JSON format as follows:\n{self.scenario_description_output_format_prompt}"
                                },
                                {
                                    "type": "text",
                                    "text": "Let's think step by step."
                                }
                            ]
            }
        ]

    async def generate_grounded_scenario_info(self, task_instruction, task_final_status, image):
        
        check_pass = await self.generate_object_dict(task_instruction, image)
        if check_pass:

            self.candidate_object = list(self.objects.keys())
            self.generate_hands_dict(task_instruction)
            self.generate_objective_dict(task_final_status)
        return check_pass


    async def generate_object_dict(self, task_instruction, image):
        self.construct_scenario_description_message(task_instruction, image)

        for _ in range(self.multi_round_dialog_times):
            check_pass = True
            res = await self.llm_invoke(self.messages)

            self.messages.append({"role": "assistant", "content": res})
            start_index = res.find("{")
            end_index = res.rfind("}") + 1 
            
            object_str = res[start_index:end_index]

            try:
                self.objects = json.loads(object_str)
            except json.JSONDecodeError:
                self.messages.append({"role": "user", "content": f"Your scenario description work's output format is wrong. Cannot be decoded into dict by json.loads(your_answer)\nPlease strictly obey the output template."})
                check_pass = False
                continue
            

            check_pass, error = self.validate_object_success()
            if check_pass is False:
                self.messages.append({"role": "user", "content": f"Your scenario description work's output is incorrect. Your errors might be:\n{error}\nPlease re-think step by step ang give the correct answer."})
                continue

            if check_pass:
                break
        
        return check_pass


    def validate_object_success(self):
        error = ""

        all_object_names = self.objects.keys()

        obj_id_list = []
        for obj_name in all_object_names:
            # name id validation
            obj_name_parts = obj_name.split('-')
            if len(obj_name_parts) != 2:
                error = f"Object names should all in the format of 'xxx-id', only one - is allowed, {obj_name} is incorrect."
                return False, error
            
            obj_id = obj_name_parts[-1]
            
            if obj_id in obj_id_list:
                error = f"Id part in object name cannot be repeated, which should be incremented one by one as a unique identifier among all objects, from 1, 2, 3, ... For example, 'phone-1', 'pen-2', 'pen-3'. {obj_name} is not allowed because '-{obj_id}' has already existed before."
                return False, error
            obj_id_list.append(obj_id)
            
            if not isinstance(self.objects[obj_name], dict) or {"type", "color", "shape", "location", "state"} != set(self.objects[obj_name].keys()):
                error = f"The value of key {obj_name} should be a dict with properties 'type', 'color', 'shape', 'location', and 'state'."
                return False, error
            
            # type validation
            obj_type = self.objects[obj_name]["type"]
            obj_type_parts = obj_type.split('/')
            if len(obj_type_parts) != 2:
                error = f"Object type should all in the format of 'superclass/subclass', {obj_type} is incorrect."
                return False, error
            obj_superclass = obj_type_parts[0]
            if obj_superclass not in ["container", "articulated", "cutlery", "food", "tool", "stationery", "cloth", "electrical", "fluid", "toy", "others"]:
                error = f"The superclass of an object must be chosen from container, articulated, cutlery, food, tool, stationery, cloth, electrical, fluid, toy, others. so {obj_superclass} is incorrect."
                return False, error
            
            obj_subclass = obj_type_parts[1]

            # color validateion
            obj_color = self.objects[obj_name]["color"]
            if obj_color not in ["red", "white", "black", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "grey", "transparent"]:
                error = f"Please choose the closest color from 'red', 'white', 'black', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'grey', 'transparent'. So {obj_color} is incorrect."
                return False, error
            
            # shape validateion
            obj_shape = self.objects[obj_name]["shape"]
            if obj_shape not in ["rectangle", "square", "circle", "triangle", "ellipse", "star", "trapezoid", "rhombus", "parallelogram", "pentagon", "hexagon", "octagon", "heart", "ring", "sector", "cuboid", "sphere", "cylinder", "cone", "cube", "Torus", "ellipsoid", "irregular"]:
                error = f"Please choose the closest shape from 'rectangle', 'square', 'circle', 'triangle', 'ellipse', 'star', 'trapezoid', 'rhombus', 'parallelogram', 'pentagon', 'hexagon', 'octagon', 'heart', 'ring', 'sector', 'cuboid', 'sphere', 'cylinder', 'cone', 'cube', 'Torus', 'ellipsoid', or 'irregular'. So {obj_shape} is incorrect."
                return False, error

            
            # location validation
            obj_location = self.objects[obj_name]["location"]
            if obj_location == obj_name or obj_location not in (list(all_object_names) + ["table", "left", "right"]):
                error = f"Location of an object can only be 'table', 'other object name', 'left', or 'right'. So {obj_location} is incorrect."
                return False, error
            
            # state validation
            obj_state = self.objects[obj_name]["state"]
            if not isinstance(obj_state, list):
                error = f"The value of state {obj_name} should be a list"
                return False, error
                
            
        return True, ""


    def generate_hands_dict(self, task_instruction):
        self.hands = {
            "left": {"location": "initial", "no_access": [], "state": []},
            "right": {"location": "initial", "no_access": [], "state": []}
        }

        constrains = self._extract_hand_constraints(task_instruction)

        for hand, cons in constrains.items():
            for description in cons:
                parts = description.split('-')
                if len(parts) == 1 or parts[0] == "all":
                    for obj_name in self.candidate_object:  # all-bottle
                        if parts[-1] in obj_name or parts[-1] in self.objects[obj_name]["type"]:
                            if obj_name not in self.hands[hand]["no_access"]:
                                self.hands[hand]["no_access"].append(obj_name)
                else:  # black-bottle / triangle-block
                    for obj_name in self.candidate_object:
                        if (parts[-1] in obj_name or parts[-1] in self.objects[obj_name]["type"]) and (parts[0] == self.objects[obj_name]["shape"] or parts[0] == self.objects[obj_name]["color"]):
                            if obj_name not in self.hands[hand]["no_access"]:
                                self.hands[hand]["no_access"].append(obj_name)
    

    def _extract_hand_constraints(self, task_instruction):
        result = {'left': [], 'right': []}
        
        hand_pattern = re.compile(
            r'<(left|right) hand>(.*?)(?=(?:<[^>]+>|$))'
        )
        
        for match in hand_pattern.finditer(task_instruction):
            hand_type = match.group(1)
            description = match.group(2).strip()
            
            if 'can approach all' in description:
                result[hand_type] = []
                continue
                
            if 'cannot approach' in description:
                items = re.findall(r'\(([^)]+)\)', description)
                cleaned_items = [item.strip() for item in items if item.strip()]
                result[hand_type] = list(set(cleaned_items))
                
        return result


    def generate_objective_dict(self, task_final_status):
        
        gorunded_located_dict = dict()
        
        for key, value in task_final_status.items():
            if key.split('-')[0] not in ["left", "right", "color", "shape"]:  # key: concerned object (ungrounded)
                located_info = value["location"]  # table / left / right / specific kind of type of object
                parts = located_info.split("-")

                # mapping ungrounded located_info to one object in scenario
                if located_info in ["table", "left", "right", "hand"]: 
                    gorunded_located_dict[located_info] = located_info
                else:  # object type, object name, shape-object, color-object
                    for obj_name in self.candidate_object:
                        concerned_located_embellish_part = parts[0]
                        concerned_located_object_part = parts[-1]

                        if concerned_located_object_part in obj_name or concerned_located_object_part in self.objects[obj_name]["type"]:
                            if len(parts) == 1:
                                gorunded_located_dict[located_info] = obj_name
                                break
                            elif concerned_located_embellish_part == self.objects[obj_name]["shape"] or concerned_located_embellish_part == self.objects[obj_name]["color"]:
                                gorunded_located_dict[located_info] = obj_name
                                break
                

        for key, value in task_final_status.items():
            if key in ["left", "right"]:
                self.objective[key] = task_final_status[key]
            else:
                parts = key.split("-")
                concerned_embellish_part = parts[0]
                concerned_object_part = parts[-1]


                located_info = value["location"]

                state_info = value["state"]

                if len(parts) == 1 or parts[0] == "all":
                    for obj_name in self.candidate_object:
                        if concerned_object_part in obj_name or concerned_object_part in self.objects[obj_name]["type"]:
                            self.objective[obj_name] = dict()
                            self.objective[obj_name]["state"] = state_info
                            self.objective[obj_name]["location"] = gorunded_located_dict[located_info]


                elif concerned_embellish_part in ["shape", "color"]:  # color-fruit: "location": "color-paper" / shape-block: "location": "shape-slot"
                    located_info_parts = value["location"].split('-')  
                    concerned_located_embellish_part = located_info_parts[0]
                    concerned_located_object_part = located_info_parts[-1]
                    assert concerned_embellish_part == concerned_located_embellish_part, "These two parts should be set same."

                    for obj_name1 in self.candidate_object:
                        if concerned_object_part in obj_name1 or concerned_object_part in self.objects[obj_name1]["type"]:  # obj1 is fruit
                            obj_name1_embellish = self.objects[obj_name1]["color"] if concerned_embellish_part == "color" else self.objects[obj_name1]["shape"]
                            for obj_name2 in self.candidate_object:
                                obj_name2_embellish = self.objects[obj_name2]["color"] if concerned_embellish_part == "color" else self.objects[obj_name2]["shape"]
                                if obj_name1_embellish == obj_name2_embellish and (concerned_located_object_part in obj_name2 or concerned_located_object_part in self.objects[obj_name2]["type"]):  # obj2 is paper
                                        self.objective[obj_name1] = dict()
                                        self.objective[obj_name1]["state"] = state_info
                                        self.objective[obj_name1]["location"] = obj_name2

                                        break
                
                else:
                    for obj_name in self.candidate_object:  # black-plug / triangle-block
                        if (concerned_object_part in obj_name or concerned_object_part in self.objects[obj_name]["type"]) and (concerned_embellish_part == self.objects[obj_name]["shape"] or concerned_embellish_part == self.objects[obj_name]["color"]):
                            self.objective[obj_name] = dict()
                            self.objective[obj_name]["state"] = state_info
                            self.objective[obj_name]["location"] = gorunded_located_dict[located_info]
        



class TaskPlannerRule(TaskSimulator):
    def __init__(self, agent_type, model_name, temperature):
        super().__init__(agent_type, model_name, temperature)

        self.stage_generation_system_prompt = read_prompt_from_file(file_path="prompts/stage_generation_system.txt")
        self.stage_generation_output_prompt = read_prompt_from_file(file_path="prompts/stage_generation_output.txt")


        self.dexterous_skill = ["hold", "align", "grasp", "place", "open", "close", "mash", "insert", "shake", "stir", "peel", "pour", "cut", "twist", "wipe"]
        
        self.stages = []
        self.plan = None
        
    def construct_stage_generation_message(self, task_instruction, annotated_image):
        self.messages = [
            {
                "role": "user",
                "content": [
                                {
                                    "type": "text",
                                    "text": self.stage_generation_system_prompt
                                },
                                {
                                    "type": "text",
                                    "text": f"The queried task instruction is: {task_instruction}"
                                },
                                {
                                    "type": "text",
                                    "text": "The observation image is:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image(annotated_image)}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": f"\n\nThe grounded `objects` dict is:\n```json{json.dumps(self.objects)}```"
                                },
                                {
                                    "type": "text",
                                    "text": f"\n\nThe grounded `hands` dict is:\n```json{json.dumps(self.hands)}```"
                                },
                                {
                                    "type": "text",
                                    "text": f"\n\nThe grounded `objective` dict is:\n```json{json.dumps(self.objective)}```"
                                },
                                {
                                    "type": "text",
                                    "text": f"Please output your answer in the format as follows: {self.stage_generation_output_prompt}"
                                },
                                {
                                    "type": "text",
                                    "text": "Let's think step by step."
                                }
                            ]
            }
        ]

    async def run_with_validation(self, task_instruction, task_final_status, image_path):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        zed_image = cv2.imread(image_path)
        width = zed_image.shape[1] // 2
        image = zed_image[:, :width]        

        # Load the ground-truth (GT) scenario grounding information
        if not Load_Grounded_Info:

            # Generate scenario grounding information based on hierarchical scenario grounding module (VLM-based + Rule-based)
            check_pass = await self.generate_grounded_scenario_info(task_instruction, task_final_status, image)

            print("objects dict:")
            formatted_json = json.dumps(self.objects, indent=4)
            print(formatted_json)
            print("hands dict:")
            formatted_json = json.dumps(self.hands, indent=4)
            print(formatted_json)
            print("objective dict:")
            formatted_json = json.dumps(self.objective, indent=4)
            print(formatted_json)
            print(check_pass)


            b = json.dumps(self.objects)
            f2 = open(os.path.join(Save_Root, "objects.json"), 'w')
            f2.write(b)
            f2.close()
            b = json.dumps(self.hands)
            f2 = open(os.path.join(Save_Root, "hands.json"), 'w')
            f2.write(b)
            f2.close()
            b = json.dumps(self.objective)
            f2 = open(os.path.join(Save_Root, "objective.json"), 'w')
            f2.write(b)
            f2.close()
            

            if check_pass == False:
                print("The first work of generating scenario description fialed. End the program.")
                exit()
        else:
            with open(os.path.join(Grounded_Root, "objects.json"), "r", encoding="utf-8") as file:
                self.objects = json.load(file)
            with open(os.path.join(Grounded_Root, "hands.json"), "r", encoding="utf-8") as file:
                self.hands = json.load(file)
            with open(os.path.join(Grounded_Root, "objective.json"), "r", encoding="utf-8") as file:
                self.objective = json.load(file)

            self.candidate_object = list(self.objects.keys())


        self.construct_stage_generation_message(task_instruction, image)
        
        check_pass = False
        for diag_ind in range(self.multi_round_dialog_times):
            res = await self.llm_invoke(self.messages)
                
            self.messages.append({"role": "assistant", "content": res})

            format_succ, format_res = self.validate_stages_format(res)
            if not format_succ:
                self.messages.append({"role": "user", "content": f"{format_res}. Your stages output format is wrong. Please strictly obey the output template."})
                check_pass = False
                continue
            if len(self.stage_list) > 2*Task_GT_Step:
                break
            print(res)
            validation_succ, validation_res = self.validate_stages_with_skill_world_model(format_res)
            print(validation_res)

            if validation_succ is False:
                self.messages.append({"role": "user", "content": f"Error note: {validation_res}\n\n\n Please strically obey all notes and re-generate a correct plan. Let's think step by step."})
                check_pass = False
                continue

            check_pass = True
            break

        print("Diag Ind: ", diag_ind+1)

        b = json.dumps(self.objects)
        f2 = open(os.path.join(Save_Root, "objects.json"), 'w')
        f2.write(b)
        f2.close()
        b = json.dumps(self.hands)
        f2 = open(os.path.join(Save_Root, "hands.json"), 'w')
        f2.write(b)
        f2.close()
        b = json.dumps(self.objective)
        f2 = open(os.path.join(Save_Root, "objective.json"), 'w')
        f2.write(b)
        f2.close()
        res_file_name = "success.txt" if check_pass else "fail.txt"
        with open(os.path.join(Save_Root, res_file_name),'w') as f:   
            for item in self.stage_list:
                f.write(f"{item}\n")         

        data_dict = {
            "feedback": diag_ind,
            "stage": len(self.stage_list) if check_pass and len(self.stage_list) <= 2*Task_GT_Step else 2*Task_GT_Step +1
        }
        b = json.dumps(data_dict)
        f2 = open(os.path.join(Save_Root, "data.json"), 'w')
        f2.write(b)
        f2.close()
        return check_pass, res
    
    
    def validate_stages_format(self, res):
        pattern = r'^Stages:\s*(?:\r?\n\s*- Stage \d+: <[^>]+> \[[^\]]+\] \([^\)]+\)\s*)+$'

        start_index = res.find("Stages:")
        if start_index == -1:
            return False, "Error"

        end_index = res.rfind(")")
        if end_index == -1:
            return False, "Error"

        stage_str = res[start_index:end_index + 1]

        if not re.fullmatch(pattern, stage_str):
            return False, "Error"
        
        lines = stage_str.strip().split('\n')

        self.stage_list = [line.split(':', 1)[1].strip() for line in lines[1:] if ':' in line] 

        success_res = True
        format_res = []
        for stage_ind, current_stage in enumerate(self.stage_list):
            success_res, parsed_res = self.parse_stage(current_stage)
            if not success_res:
                return False, f"(Error in Stage {stage_ind}) " + parsed_res
            format_res.append(parsed_res)

        return True, format_res


    def parse_stage(self, stage_str):
        angle_pattern = r'<([^>]+)>'
        square_pattern = r'\[([^\]]+)\]'
        paren_pattern = r'\(([^)]+)\)'

        angle_match = re.search(angle_pattern, stage_str)
        square_match = re.search(square_pattern, stage_str)
        paren_match = re.search(paren_pattern, stage_str)

        angle = angle_match.group(1).strip() if angle_match else ""
        square = square_match.group(1).strip() if square_match else ""
        paren = paren_match.group(1).strip() if paren_match else ""

        angle = angle.split(" ")[0]
        format_id = 0
        if square == "approach":
            format_id = 1
            if angle not in ["left", "right"]:
                error = f"For [approach], only support <left hand> / <right hand>, not <{angle} hand>."
                return False, error
            
            app_target_name_list = ["initial", "table"] + self.candidate_object
            if paren not in app_target_name_list:
                error = f"{stage_str}, app_target_name must be in the set of {app_target_name_list}."
                return False, error

        elif square == "twist":
            format_id = 2
            if angle not in ["left", "right"]:
                error = f"For [approach], only support <left hand> / <right hand>, not <{angle} hand>."
                return False, error
            
            parts = paren.split('+')
            if len(parts) != 3:
                error = f"For <hand_ind hand> [{square}] (xxx), xxx must be in the format of tool_name+dex_base_name+dex_target_name, so {paren} is incorrect."
                return False, error
            
            tool_name = parts[0]
            dex_base_name = parts[1]
            dex_target_name = parts[2]
            tool_name_list = ["left", "right"] + self.candidate_object
            dex_base_name_list = self.candidate_object
            dex_target_name_list = self.candidate_object


            if tool_name not in tool_name_list:
                error = f"{stage_str}, tool_name must be in the set of {tool_name_list}, so {tool_name} is incorrect."
                return False, error
            
            if dex_base_name not in dex_base_name_list:
                error = f"{stage_str}, dex_base_name must be in the set of {dex_base_name_list}, so {dex_base_name} is incorrect."
                return False, error
            
            if dex_target_name not in dex_target_name_list:
                error = f"{stage_str}, dex_target_name must be in the set of {dex_target_name_list}, so {dex_target_name} is incorrect."
                return False, error


        elif square in self.dexterous_skill:
            format_id = 2
            if angle not in ["left", "right"]:
                error = f"For [approach], only support <left hand> / <right hand>, not <{angle} hand>."
                return False, error
            
            parts = paren.split('+')
            if len(parts) != 2:
                error = f"For <hand_ind hand> [{square}] (xxx), xxx must be in the format of tool_name+dex_target_name, so {paren} is incorrect."
                return False, error

            tool_name = parts[0]
            dex_target_name = parts[1]
            tool_name_list = ["left", "right"] + self.candidate_object
            dex_target_name_list = ["table"] + self.candidate_object

            if square == "hold" and tool_name != angle:
                error = f"For <hand_ind hand> [hold] (tool_name+dex_target_name), tool_name must be set as {angle} so {tool_name} is incorrect."
                return False, error
            
            if square == "align" and tool_name not in self.candidate_object:
                error = f"{stage_str}, tool_name must be set of {self.candidate_object} so {tool_name} is incorrect."
                return False, error
            
            if square == "align" and dex_target_name not in self.candidate_object:
                error = f"{stage_str}, tool_name must be set of {self.candidate_object} so {tool_name} is incorrect."
                return False, error

            if square in ["place", "insert", "stir", "pour", "wipe"] and tool_name not in self.candidate_object:
                error = f"{stage_str}, tool_name must be set of {self.candidate_object} so {tool_name} is incorrect."
                return False, error

            if square not in ["wipe", "place"] and dex_target_name == "table":
                error = f"{stage_str}, dex_target_name must be set of {self.candidate_object} so {dex_target_name} is incorrect."
                return False, error
                


            if tool_name not in tool_name_list:
                error = f"{stage_str}, tool_name must be in the set of {tool_name_list}, so {tool_name} is incorrect."
                return False, error
            
            if dex_target_name not in dex_target_name_list:
                error = f"{stage_str}, dex_target_name must be in the set of {dex_target_name_list}, so {dex_target_name} is incorrect."
                return False, error


        elif square == "move":
            format_id = 3
            if angle != "both":
                error = f"For [{square}], only support <both hands>, not <{angle} hands>."
                return False, error
            
            parts = paren.split('+')
            if len(parts) != 2:
                error = f"For <both hands> [{square}] (xxx), xxx must be in the format of obj_inhand_name+move_target_name, so {paren} is incorrect."
                return False, error

            obj_inhand_name = parts[0]
            move_target_name = parts[1]
            obj_inhand_name_list = self.candidate_object
            move_target_name_list = ["table", "transport"] + self.candidate_object

            if obj_inhand_name not in obj_inhand_name_list:
                error = f"For <both hands> [{square}] (obj_inhand_name+move_target_name), obj_inhand_name must be in the set of {obj_inhand_name_list}, so {obj_inhand_name} is incorrect."
                return False, error
            
            if move_target_name not in move_target_name_list:
                error = f"For <both hands> [{square}] (obj_inhand_name+move_target_name), move_target_name must be in the set of {move_target_name_list}, so {move_target_name} is incorrect."
                return False, error
            
        elif square == "handover":
            format_id = 4
            if angle != "both":
                error = f"For [{square}], only support <both hands>, not <{angle} hands>."
                return False, error
            
            parts = paren.split('+')
            if len(parts) != 2:
                error = f"For <both hands> [{square}] (xxx), xxx must be in the format of active_name+passive_name, so {paren} is incorrect."
                return False, error

            active_name = parts[0]
            passive_name = parts[1]
            active_name_list = self.candidate_object
            passive_name_list = ["left", "right"]

            if active_name not in active_name_list:
                error = f"For <both hands> [{square}] (active_name+passive_name), active_name must be in the set of {active_name_list}, so {active_name} is incorrect."
                return False, error
            
            if passive_name not in passive_name_list:
                error = f"For <both hands> [{square}] (active_name+passive_name), passive_name must be in the set of {passive_name_list}, so {passive_name} is incorrect."
                return False, error
        
        if format_id == 0:
            return False, f"[{square}] is not supported"
        else:
            return True, {
                            "hand": angle,
                            "skill": square,
                            "object": paren,
                            "id": format_id
                        }



    def validate_stages_with_skill_world_model(self, parsed_stage_list):
        """
           If error, only feedback the error itself not the whole reasoning process from the beginning stage to the error stage.
        """
        # managed status for objects
        objects_copy = copy.deepcopy(self.objects)
        self.object_location_status = dict()
        self.object_state_status = dict()
        for obj_name, obj_value in objects_copy.items():
            self.object_location_status[obj_name] = obj_value["location"]
            self.object_state_status[obj_name] = obj_value["state"]
        

        # managed status for hands
        hands_copy = copy.deepcopy(self.hands)
        self.hands_no_access_status = dict()
        self.hands_state_status = dict()
        self.hands_location_status = dict()
        self.hands_inhand_status = dict()
        for hand_ind, hand_value in hands_copy.items():
            self.hands_no_access_status[hand_ind] = hand_value["no_access"]
            self.hands_state_status[hand_ind] = hand_value["state"]
            self.hands_location_status[hand_ind] = hand_value["location"]
            self.hands_inhand_status[hand_ind] = None


        self.hold_obj_name = None
        self.hold_hand_name = None
        self.aligned_obj_name = None
        # Check whether each stage can be excuted
        for stage_ind, current_stage in enumerate(parsed_stage_list):
            which_skill = current_stage["skill"]

            if which_skill == "move":
                res, info = self._modification_function_move(stage_ind, current_stage)
            elif which_skill == "handover":
                res, info = self._modification_function_handover(stage_ind, current_stage)
            elif which_skill == "align":
                res, info = self._modification_function_align(stage_ind, current_stage)
            elif which_skill == "approach":
                res, info = self._modification_function_approach(stage_ind, current_stage)
            elif which_skill == "grasp":
                res, info = self._modification_function_grasp(stage_ind, current_stage)
            elif which_skill == "place":
                res, info = self._modification_function_place(stage_ind, current_stage)
            elif which_skill == "insert":
                res, info = self._modification_function_insert(stage_ind, current_stage)
            elif which_skill == "pour":
                res, info = self._modification_function_pour(stage_ind, current_stage)
            elif which_skill == "twist":
                res, info = self._modification_function_twist(stage_ind, current_stage)
            elif which_skill == "peel":
                res, info = self._modification_function_peel(stage_ind, current_stage)
            elif which_skill == "shake":
                res, info = self._modification_function_shake(stage_ind, current_stage)
            elif which_skill == "open":
                res, info = self._modification_function_open(stage_ind, current_stage)
            elif which_skill == "close":
                res, info = self._modification_function_close(stage_ind, current_stage)
            elif which_skill == "hold":
                res, info = self._modification_function_hold(stage_ind, current_stage)
            else:
                res, info = self._modification_function_other(stage_ind, current_stage)

            if res is False:
                return False, info

        # Check whether final status fits the objective

        res, info = self._check_whether_fianl_status_meet()
        return res, info



    def _check_whether_fianl_status_meet(self):
        
        elsewhere = False
        for hand_ind in ["left", "right"]:
            if "initial" not in self.hands_location_status[hand_ind]:
                elsewhere = True
        
        if elsewhere:
            self.hands_location_status["left"] = "elsewhere"
            self.hands_location_status["right"] = "elsewhere"
        else:
            self.hands_location_status["left"] = "initial"
            self.hands_location_status["right"] = "initial"

                
        for hand_ind in ["left", "right"]:
            target = self.objective[hand_ind]["location"]
            curr = self.hands_location_status[hand_ind]
            if target != curr:
                return False, f"After all stages finished, the location field of {hand_ind} in modified `hands` dict is wrong, it should be {target} but now is {curr}."
            
            target = self.objective[hand_ind]["state"]
            for s in target:
                if s not in self.hands_state_status[hand_ind]:
                    return False, f"After all stages finished, the state field of {hand_ind} in modified `hands` dict is wrong, it should contain {s} state but now it is missing."

        for concerned_obj,concerned_value in self.objective.items():
            if concerned_obj not in ["left", "right"]:
                target = concerned_value["location"]
                curr = self.object_location_status[concerned_obj]
                if target == "hand" and curr not in ["left", "right"]:
                    return False, f"After all stages finished, the location field of {concerned_obj} in modified `objects` dict is wrong, it should be left or right but now is {curr}."
                if target != "hand" and target != curr:
                    return False, f"After all stages finished, the location field of {concerned_obj} in modified `objects` dict is wrong, it should be {target} but now is {curr}."
                target = concerned_value["state"]
                for s in target:
                    if s not in self.object_state_status[concerned_obj]:
                        return False, f"After all stages finished, the state field of {concerned_obj} in modified `objects` dict is wrong, it should contain {s} state but now it is missing."
        return True, ""

    def _modification_function_move(self, stage_ind, current_stage):
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        obj_inhand_name, move_target_name = which_object.split('+')

        for hand_ind in ["left", "right"]:
            if self.hands_inhand_status[hand_ind] != obj_inhand_name:
                return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {hand_ind} hand does not grasp {obj_inhand_name} in the modified `hands` dict. If [{which_skill}] is necessary, please refer to the provided sequential stage block for more details."
            
            if move_target_name in self.hands_no_access_status[hand_ind]:
                return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {move_target_name} is in the no_access list of {hand_ind} in the modified `hands` dict, which means {hand_ind} cannot reach {move_target_name} so that this skill is failed. If [{which_skill}] is necessary, please refer to the provided sequential stage block for more details."

        self.hands_location_status["left"] = move_target_name
        self.hands_location_status["right"] = move_target_name
        if self.hold_hand_name is not None:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None

        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""


    def _modification_function_handover(self, stage_ind, current_stage):
        which_object = current_stage["object"]
        active_name, passive_name = which_object.split('+')

        hand_ind = self.object_location_status[active_name]
        if hand_ind == "left":
            the_other_ind = "right"
        else:
            the_other_ind = "left"

        if hand_ind not in ["left", "right"]:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because the location of {active_name} in modified `objects` dict is not in hands. It is currently on {hand_ind}, so you cannot handover {active_name} using your hands. If [handover] is necessary, please refer to the provided sequential stage block for more details."
        
        if hand_ind == "left":
            the_other_ind = "right"
        else:
            the_other_ind = "left"

        if passive_name != the_other_ind:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because passive_name must be {the_other_ind} if you want to handover {active_name} from {hand_ind} to {the_other_ind}. If [handover] is necessary, please refer to the provided sequential stage block for more details."
        
        if self.hands_inhand_status[passive_name] is not None:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {passive_name} is not empty (inhand field is not None) according to the modified `hands` dict. If [handover] is necessary, please refer to the provided sequential stage block for more details."

        
        self.hands_inhand_status[hand_ind] = None
        self.hands_inhand_status[the_other_ind] = active_name
        self.hands_location_status[hand_ind] = "transport"
        self.hands_location_status[the_other_ind] = "transport"
        self.object_location_status[active_name] = the_other_ind

        self._update_no_access_list(active_name, the_other_ind)

        if self.hold_hand_name is not None:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None

        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""
    


    def _modification_function_align(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        tool_name, dex_target_name = which_object.split('+')

        if self.object_location_status[tool_name] not in ["left", "right"]:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because the location of {tool_name} in modified `objects` dict is not in hands. It is currently on {self.object_location_status[tool_name]}, so you cannot align {tool_name} it with dex_target_name. If [align] is necessary, please refer to the provided sequential stage block for more details."
        
        hand_ind = self.object_location_status[tool_name]
        if self.hands_location_status[hand_ind] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {hand_ind} did not approach to {dex_target_name} before so you cannot align {tool_name} with {dex_target_name}. Please refer to the provided sequential stage block for more details."
        
        if "aligned" not in self.object_state_status[tool_name]:
            self.object_state_status[tool_name].append("aligned")

        self.aligned_obj_name = tool_name

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
            

        return True, ""
    

    def _modification_function_approach(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        app_target_name = current_stage["object"]

        if app_target_name in self.hands_no_access_status[which_hand]:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {app_target_name} is in the no_access list of {which_hand} in modified `hands` dict. If manipulating this object is necessary, try to use a hand that can access to it or you may need to handover it."

        self.hands_location_status[which_hand] = app_target_name
        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None
        return True, ""


    def _modification_function_grasp(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')

        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before. If grasp it is necessary, try to approach it and refer to the provided sequential stage block for more details."
            

        if tool_name in ["left", "right"] and self.hands_inhand_status[which_hand] is not None:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} has already grasped an object before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."
        
        
        self.object_location_status[dex_target_name] = which_hand
        self.hands_inhand_status[which_hand] = dex_target_name

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None
        return True, ""

    def _update_no_access_list(self, tool_name, dex_target_name):
        for hand_ind in ["left", "right"]:
            if tool_name not in self.hands_no_access_status[hand_ind] and dex_target_name in self.hands_no_access_status[hand_ind]:
                self.hands_no_access_status[hand_ind].append(tool_name)
            if tool_name in self.hands_no_access_status[hand_ind] and dex_target_name not in self.hands_no_access_status[hand_ind]:
                self.hands_no_access_status[hand_ind].remove(tool_name)
    
    def _modification_function_place(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        
        tool_name, dex_target_name = which_object.split('+')

        if self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} according to the modified `hands` dict so you cannot place it. Please refer to the provided sequential stage block for more details."
        
        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot place {tool_name} on {dex_target_name}. Please refer to the provided sequential stage block for more details."
        

        self.object_location_status[tool_name] = dex_target_name
        self.hands_inhand_status[which_hand] = None
        self._update_no_access_list(tool_name, dex_target_name)

        if dex_target_name in self.candidate_object and "container" in self.objects[dex_target_name]["type"]:
            self.object_state_status[dex_target_name].append(f"with+{tool_name}")



        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None
        return True, ""
 
    def _modification_function_insert(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')

        
        if self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} according to the modified `hands` dict so you cannot {which_skill} it. Please refer to the provided sequential stage block for more details."
        
        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {tool_name} on {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if self.aligned_obj_name is None:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} did not [align] before. Please refer to the provided sequential stage block for more details."
            

        if "fixed" not in self.object_state_status[dex_target_name] and "held" not in self.object_state_status[dex_target_name]:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because the state list of {dex_target_name} dose not contain fixed and held state according to the modified `objects` dict so it is unstable to {which_skill}. Please refer to the provided sequential stage block for more details."



        self.object_location_status[tool_name] = dex_target_name
        self.hands_inhand_status[which_hand] = None
        self._update_no_access_list(tool_name, dex_target_name)

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""
    
    def _modification_function_twist(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_base_name, dex_target_name = which_object.split('+')

        if tool_name not in ["left", "right"] and self.object_location_status[dex_base_name] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {dex_base_name} is not located on {dex_target_name} according to the modified `objects` dict so you cannot using {tool_name} to {which_skill} {dex_base_name} into {dex_target_name}. To do so, you need to [insert] (dex_base_name+dex_target_name) first. Please refer to the provided sequential stage block for more details."
        
        if tool_name in ["left", "right"] and self.hands_inhand_status[which_hand] != dex_base_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {dex_base_name} is not in {which_hand} according to the modified `hands` dict so you cannot {which_skill} it into {dex_target_name}. Please refer to the provided sequential stage block for more details."
        
        
        if tool_name not in ["left", "right"] and self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} according to the modified `hands` dict so you cannot using it to {which_skill} {dex_base_name} into {dex_target_name}. Please refer to the provided sequential stage block for more details."
        
        if tool_name in ["left", "right"] and self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot using {tool_name} {which_skill} {dex_base_name} into {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if tool_name not in ["left", "right"] and self.hands_location_status[which_hand] != dex_base_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_base_name} before so you cannot using {tool_name} {which_skill} {dex_base_name} into {dex_target_name}. Please refer to the provided sequential stage block for more details."

            
        if "fixed" not in self.object_state_status[dex_target_name] and "held" not in self.object_state_status[dex_target_name]:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because the state list of {dex_target_name} dose not contain fixed and held state according to the modified `objects` dict so it is unstable to {which_skill}. Please refer to the provided sequential stage block for more details."



        self.object_location_status[dex_base_name] = dex_target_name

        self._update_no_access_list(dex_base_name, dex_target_name)

        if self.hands_inhand_status[which_hand] == dex_base_name:
            self.hands_inhand_status[which_hand] = None
        self.object_state_status[dex_base_name].append("twisted")

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""
    
    def _modification_function_peel(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')


        if tool_name not in ["left", "right"] and self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} according to the modified `hands` dict so you cannot use it to {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if tool_name in ["left", "right"] and self.hands_inhand_status[which_hand] is not None:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} has already grasped an object before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."


        if "fixed" not in self.object_state_status[dex_target_name] and "held" not in self.object_state_status[dex_target_name]:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because the state list of {dex_target_name} dose not contain fixed and held according to the modified `objects` dict so it is unstable to {which_skill}. Please refer to the provided sequential stage block for more details."


        self.object_state_status[dex_target_name].append("peeled")
        

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""

    def _modification_function_shake(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')


        if tool_name not in ["left", "right"] and self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} according to the modified `hands` dict so you cannot use it to {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if tool_name in ["left", "right"] and self.hands_inhand_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not grasp {dex_target_name} before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."



        self.object_state_status[dex_target_name].append("shaken")
        

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""
    
    def _modification_function_pour(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')

        
        if self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} according to the modified `hands` dict so you cannot {which_skill} it. Please refer to the provided sequential stage block for more details."
        
        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {tool_name} to {dex_target_name}. Please refer to the provided sequential stage block for more details."


        for s in self.object_state_status[tool_name]:
            if "with+" in s and s not in self.object_state_status[dex_target_name]:
                self.object_state_status[dex_target_name].append(s)

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""
 
    def _modification_function_open(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')


        if tool_name in ["left", "right"] and self.hands_inhand_status[which_hand] is not None:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} has already grasped an object before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if tool_name in ["left", "right"] and self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if tool_name not in ["left", "right"] and self.hands_location_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {tool_name} before so you cannot {which_skill} {tool_name}. Please refer to the provided sequential stage block for more details."


        if "closed" in self.object_state_status[dex_target_name]:
            self.object_state_status[dex_target_name].remove("closed")
        if "opened" not in self.object_state_status[dex_target_name]:
            self.object_state_status[dex_target_name].append("opened")
        
        if tool_name not in ["left", "right"]:
            self.object_location_status[tool_name] = which_hand
            self.hands_inhand_status[which_hand] = tool_name

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""
    

    def _modification_function_close(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')


        if tool_name not in ["left", "right"] and self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} so you cannot use it to {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."
    

        if tool_name in ["left", "right"] and self.hands_inhand_status[which_hand] is not None:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} has already grasped an object before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."


        if "opened" in self.object_state_status[dex_target_name]:
            self.object_state_status[dex_target_name].remove("opened")
        if "closed" not in self.object_state_status[dex_target_name]:
            self.object_state_status[dex_target_name].append("closed")

        if tool_name not in ["left", "right"]:
            self.object_location_status[tool_name] = dex_target_name
            self.hands_inhand_status[which_hand] = None
            
            self._update_no_access_list(tool_name, dex_target_name)

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""
    

    def _modification_function_hold(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')

        if tool_name in ["left", "right"] and self.hands_inhand_status[which_hand] is not None:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} has already grasped an object before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."
        if "held" not in self.object_state_status[dex_target_name]:
            self.object_state_status[dex_target_name].append("held")
        self.hold_obj_name = dex_target_name
        self.hold_hand_name = tool_name

        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None
        return True, ""

    def _modification_function_other(self, stage_ind, current_stage):
        which_hand = current_stage["hand"]
        which_object = current_stage["object"]
        which_skill = current_stage["skill"]
        
        tool_name, dex_target_name = which_object.split('+')


        if tool_name not in ["left", "right"] and self.hands_inhand_status[which_hand] != tool_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {tool_name} is not in {which_hand} according to the modified `hands` dict so you cannot use it to {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."
    

        if self.hands_location_status[which_hand] != dex_target_name:
            return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because {which_hand} did not approach to {dex_target_name} before so you cannot {which_skill} {dex_target_name}. Please refer to the provided sequential stage block for more details."

        if which_skill in ["mash", "cut", "stir", "wipe"]:
            if "fixed" not in self.object_state_status[dex_target_name] and "held" not in self.object_state_status[dex_target_name]:
                return False, f"Cannot execute Stage {stage_ind+1}: {self.stage_list[stage_ind]} because the state list of {dex_target_name} dose not contain fixed and held according to the modified `objects` dict so it is unstable to {which_skill}. Please refer to the provided sequential stage block for more details."

        if which_skill == "cut":
            self.object_state_status[dex_target_name].append("cut")
        elif which_skill == "mash":
            self.object_state_status[dex_target_name].append("mashed")
        elif which_skill == "stir":
            self.object_state_status[dex_target_name].append("stirred")
        elif which_skill == "wipe":
            self.object_state_status[dex_target_name].append("wiped")
        

        if self.hold_hand_name == which_hand:
            self.object_state_status[self.hold_obj_name].remove("held")
            self.hold_obj_name = None
            self.hold_hand_name = None
        
        if self.aligned_obj_name is not None:
            self.object_state_status[self.aligned_obj_name].remove("aligned")
            self.aligned_obj_name = None

        return True, ""



def validate_task_instruction(task_instruction):
    """
    Check whether the task instruction obeys the format rules
    """
    # Regular expression template for three parts in the task instruction
    workspace_pattern = re.compile(
    r'## Workspace: Notice that (the)?\s*<(left|right) hand> '
    r'(?:cannot approach (\([^)]+\)(?:,\s*|,\s*and\s*| \s*and\s*)?)+(?: on the table)?|can approach all objects(?: on the table)?)'
    r'(?:,?\s*and\s*(the)? <(?!\1)(left|right) hand> '
    r'(?:cannot approach (\([^)]+\)(?:,\s*|,\s*and\s*| \s*and\s*)?)+(?: on the table)?|can approach all objects(?: on the table)?))?\..*$'
)
    objective_pattern = r"## Objective: .+\."
    completion_pattern = r"## Post-task action: After completing the task, please .+\."

    if not re.search(workspace_pattern, task_instruction):
        print("Defined workspace does not match the template.")
        return False
    if not re.search(objective_pattern, task_instruction):
        print("Objective does not match the template.")
        return False
    if not re.search(completion_pattern, task_instruction):
        print("Completion instruction does not match the template.")
        return False

    print("Task instruction is valid.")
    return True

def run_with_TaskPlannerRule(task_instruction, task_final_status, image_path):
    check_input = validate_task_instruction(task_instruction)
    
    if check_input:
        agent_type = "openai"
        model_name = "gpt-4o"
        tp = TaskPlannerRule(agent_type, model_name, temperature=0.2)
        check_pass, res = asyncio.run(tp.run_with_validation(task_instruction, task_final_status, image_path))
        print("---------------------------------------------")
        print("Check pass", check_pass)
        print("Result:\n", res)


if __name__ == "__main__":

    Root = "exp/OBiMan-Bench"
    level = "middle.txt"


    with open(os.path.join(Root, level), "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            
            
            if len(parts) == 4:
                subdir = f"{parts[0]} {parts[1]}"
                Task_Name = parts[2] 
                Task_GT_Step = int(parts[3])


                OBiMan_Bench_Root = os.path.join(Root, subdir)
                

                Grounded_Root = os.path.join(OBiMan_Bench_Root, Task_Name, "GT")
                Save_Root = os.path.join(OBiMan_Bench_Root, Task_Name, "GT_w_image")
                if not os.path.exists(Grounded_Root):
                    print(f"{Grounded_Root} does not exist.")
                    exit()
                    
                if not os.path.exists(Save_Root):
                    os.mkdir(Save_Root)
                else:
                    shutil.rmtree(Save_Root)
                    os.mkdir(Save_Root)


                task_root = os.path.join(OBiMan_Bench_Root, Task_Name)
                print("=================================================")
                print(task_root)

                with open(os.path.join(task_root, "instruction.txt"), "r", encoding="utf-8") as file:
                    task_instruction = file.read()


                with open(os.path.join(task_root, "objective.json"), "r", encoding="utf-8") as json_file:
                    task_final_status = json.load(json_file)

                image_path = os.path.join(task_root, "robot.jpg")

                run_with_TaskPlannerRule(task_instruction, task_final_status, image_path)