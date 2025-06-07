import re
import os
from pathlib import Path
from ParseModelSystem import ModelSys
from AssemblyFileCreation import AssemblyFileCreation
from utility_fncs import UtilityFunctions

class SysToAsm:
    """
    Class to convert system definitions to assembly files.
    """

    def __init__(self):
        """
        Initialize the SysToAsm class with utility, assembly, and model system helpers.
        """
        self.utfncs = UtilityFunctions()
        self.asm_funcs = AssemblyFileCreation()
        self.model_sys = ModelSys()

    def __del__(self):
        """
        Destructor for cleanup if needed.
        """
        pass

    def nu_convert_system_to_assemblies(self, system_name, system_info, output_dir, append_dicts={}):
        """
        Convert a single system definition to assembly files.

        Parameters:
        -----------
            system_name (str): The name of the system to convert.
            system_info (dict): Dictionary containing system definitions.
            output_dir (str): Directory to save the output files.
            append_dicts (dict): Additional define blocks to append.

        Returns:
        --------
            None
        """

        system_definition = self.utfncs.extract_system_definition(system_info, system_name)

        system_definition, sys_args = self.utfncs.get_sys_args_definition(system_definition)
        
        # Define paths for assembly definition and data files
        definition_file_path = os.path.join(output_dir, 'defn', f"{system_name}.mdl")
        data_file_path = os.path.join(output_dir, 'data', f"{system_name}_data.mdl")

        assembly_definition_content, assembly_data_content = self.utfncs.prepare_assembly_files(system_name,sys_args)
        
        # Regex pattern to capture all *System statements with optional spaces near curved brackets
        pattern = self.model_sys.system_pattern
        
        # Find all matches in the content
        matches = re.findall(pattern, system_definition)
        for match in matches:
            match = [m.strip() for m in match]
            
            # replace only the systems that are to be converted to assembly
            if match[2] in self.model_sys.systems_to_ignore:
                assembly_definition_content = self.model_sys.sys_includes[match[2]] + assembly_definition_content
                continue
            else:
                system_definition = re.sub(self.model_sys.getSpecificSystem(match[0]), '', system_definition)
                assembly_data_content += self.utfncs.get_sys_to_assembly_data(match[0], match[1], match[3], f"{match[2]}_data.mdl")
                assembly_definition_content += self.utfncs.get_sys_to_assembly_definition(match[0], match[2], match[1], match[3])

        system_definition += '*EndDefine()\n'

        for define_type, define_type_dict in append_dicts.items():
            system_definition = self.utfncs.append_datasets(system_definition, system_info[system_name][define_type], define_type_dict)

        assembly_definition_content += system_definition
        assembly_data_content += ''
        # End the assembly files
        assembly_definition_content += "*EndDefinitionFile()\n"
        assembly_data_content += "*EndDataFile()\n"

        self.utfncs.write_assembly_files(definition_file_path, assembly_definition_content, data_file_path, assembly_data_content)

    def execute_sys_2_asm(self, input_file_path, output_dir):
        """
        Main function to convert all systems in the input file to assembly files.

        Parameters:
        -----------
            input_file_path (str): Path to the input .mdl file.
            output_dir (str): Directory to save the converted assembly files.

        Returns:
        --------
            None
        """

        with self.utfncs.custom_open(input_file_path, 'r') as file:
            file_content = file.read()
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'defn'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

        output_file = os.path.join(output_dir, Path(input_file_path).stem + '_converted_assembly.mdl')

        system_info = self.model_sys.extract_system_info(file_content)

        system_info = self.model_sys.extract_define_blocks(file_content, '*DefineSystem', 'EndDefine')

        # system_info = getAddnDefinitionInclude(system_info)
        # Get All the necessary data separately
        self.model_sys.add_define_block(system_info, file_content, 'DataSet', key=2)
        define_dataset_map = self.model_sys.extract_define_blocks(file_content, 'DefineDataSet', 'EndDefine')
        
        define_gra_map = self.model_sys.extract_define_blocks(file_content, 'DefineGraphic', 'EndDefine')
        self.model_sys.add_define_block(system_info, file_content, 'Graphic', reqdlist=define_gra_map.keys(), key=2)
        
        self.model_sys.add_define_block(system_info, file_content, 'Form', key=2)
        define_form_map = self.model_sys.extract_define_blocks(file_content, 'DefineForm', 'EndDefine')
        self.model_sys.add_define_block(system_info, file_content, 'Template', key=3)
        define_template_map = self.model_sys.extract_define_blocks(file_content, 'DefineTemplate', 'EndDefine')
        begin_context_map = self.model_sys.parse_context(file_content)
        
        sys_defn_ins = self.model_sys.get_defn_instance_map('System', file_content)
        items_to_del = self.model_sys.clean_up_context(begin_context_map, sys_defn_ins, [])

        dict_map = {
            'DataSet': define_dataset_map,
            'Graphic': define_gra_map,
            'Form': define_form_map,
            'Template': define_template_map
        }

        for system_name in system_info.keys():
            self.nu_convert_system_to_assemblies(system_name, system_info, output_dir, dict_map)

        begin_context_map_updated = self.model_sys.update_values_with_keys(begin_context_map, items_to_del, file_content, system_info)
        updated_file_content = self.asm_funcs.update_file_content(begin_context_map, file_content)
        assembly_data = self.asm_funcs.get_input_assembly_file(updated_file_content, system_info)
        updated_file_content = self.asm_funcs.update_assembly_file(
            begin_context_map,
            updated_file_content,
            begin_context_map_updated,
            assembly_data,
            items_to_del,
            system_info
        )

        with self.utfncs.custom_open(output_file, 'w') as file:
            file.write(updated_file_content)
        print(f'System {input_file_path} converted to assembly')
        return


# input_file_path = 'C:/Altair_Installs/2025.0.0.22 fotisMVSMOV-LOCORI/hwdesktop/hw/mdl/autoentities/definitions_mdl/autoTireDeformgrasub.mdl'
input_file_path = 'D:/Projects/Jira/mvsma-2625_SystemToAssembly/Sedan_1.mdl'
output_dir = 'D:/Projects/Jira/mvsma-2625_SystemToAssembly/assemblies_test_sedan'
SysToAsm().execute_sys_2_asm(input_file_path, output_dir)
print("Conversion completed successfully.")