#!/usr/bin/env python3
"""
Add Box Feet to Genesis Skeleton XML

This script modifies the Genesis skeleton XML to add LocoMujoco-style box feet
for improved trajectory following stability.

Based on LocoMujoco implementation:
https://github.com/robfiras/loco-mujoco/blob/131c1e7bfb4c3d20ffc82c0368886fe708a5964b/loco_mujoco/environments/humanoids/base_skeleton.py#L220
"""

import xml.etree.ElementTree as ET
import os
from typing import Optional


def add_box_feet_to_xml(input_xml_path: str, output_xml_path: Optional[str] = None) -> bool:
    """
    Add box feet geometries to Genesis skeleton XML
    
    Args:
        input_xml_path: Path to input XML file
        output_xml_path: Path to output XML file (default: adds '_box_feet' suffix)
        
    Returns:
        Success status
    """
    
    if output_xml_path is None:
        base_name = os.path.splitext(input_xml_path)[0]
        output_xml_path = f"{base_name}_box_feet.xml"
    
    print(f"Adding box feet to {input_xml_path}")
    print(f"Output will be saved to {output_xml_path}")
    
    try:
        # Parse the XML
        tree = ET.parse(input_xml_path)
        root = tree.getroot()
        
        # LocoMujoco box feet parameters (scaled)
        scaling = 1.06552
        box_size = [dim * scaling for dim in [0.112, 0.03, 0.05]]  # [length, height, width]
        box_pos = [pos * scaling for pos in [-0.09, 0.019, 0.0]]   # Position relative to toe
        
        print(f"Box dimensions: {box_size}")
        print(f"Box position: {box_pos}")
        
        # Find toe bodies and add box geometries
        toe_configs = [
            ("toes_l", [0, 0.15, 0]),  # Left foot with rotation
            ("toes_r", [0, -0.15, 0])  # Right foot with rotation  
        ]
        
        boxes_added = 0
        original_geoms_disabled = 0
        
        # Search through all body elements
        for body in root.iter('body'):
            body_name = body.get('name', '')
            
            for toe_name, rotation in toe_configs:
                if body_name == toe_name:
                    print(f"Found toe body: {toe_name}")
                    
                    # Add box geometry to this toe body
                    box_geom = ET.SubElement(body, 'geom')
                    box_geom.set('name', f'foot_box_{toe_name}')
                    box_geom.set('type', 'box')
                    box_geom.set('size', f'{box_size[0]:.4f} {box_size[1]:.4f} {box_size[2]:.4f}')
                    box_geom.set('pos', f'{box_pos[0]:.4f} {box_pos[1]:.4f} {box_pos[2]:.4f}')
                    box_geom.set('euler', f'{rotation[0]} {rotation[1]} {rotation[2]}')
                    box_geom.set('rgba', '0.5 0.5 0.5 0.8')
                    box_geom.set('condim', '3')
                    box_geom.set('class', 'collision')
                    
                    boxes_added += 1
                    print(f"  ✅ Added box geometry to {toe_name}")
        
        # Disable collision for original foot geometries
        original_foot_geoms = ["r_foot_col", "r_bofoot_col", "l_foot_col", "l_bofoot_col"]
        
        for geom in root.iter('geom'):
            geom_name = geom.get('name', '')
            if geom_name in original_foot_geoms:
                # Disable collision by setting contype and conaffinity to 0
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
                original_geoms_disabled += 1
                print(f"  ✅ Disabled collision for {geom_name}")
        
        # Save the modified XML
        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
        
        print(f"\n✅ Box feet XML modification completed:")
        print(f"   - {boxes_added} box geometries added")
        print(f"   - {original_geoms_disabled} original foot collision geometries disabled")
        print(f"   - Output saved to: {output_xml_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to add box feet: {e}")
        return False


def main():
    """Main function to add box feet to skeleton XML"""
    
    # Default XML file path
    skeleton_dir = os.path.dirname(os.path.abspath(__file__))
    input_xml = os.path.join(skeleton_dir, "revised_genesis_skeleton.xml")
    
    if not os.path.exists(input_xml):
        print(f"❌ XML file not found: {input_xml}")
        print("Available XML files in skeleton directory:")
        for file in os.listdir(skeleton_dir):
            if file.endswith('.xml'):
                print(f"  - {file}")
        return
    
    # Add box feet to the XML
    success = add_box_feet_to_xml(input_xml)
    
    if success:
        output_file = os.path.splitext(input_xml)[0] + "_box_feet.xml"
        print(f"\n🎯 Next steps:")
        print(f"1. Update skeleton_humanoid.py to use: {os.path.basename(output_file)}")
        print(f"2. Test trajectory following with box feet enabled")
        print(f"3. Compare stability with and without box feet")
    else:
        print(f"\n❌ Box feet addition failed")


if __name__ == "__main__":
    main()