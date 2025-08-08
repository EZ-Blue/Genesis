#!/usr/bin/env python3
"""
Fix Box Feet Implementation for Genesis Skeleton

Creates a corrected box feet XML that exactly matches LocoMujoco's implementation:
- Proper box dimensions and positioning
- Correct collision property settings
- Visual appearance matching LocoMujoco
- Both feet should have visible box geometries
"""

import xml.etree.ElementTree as ET
import os
import numpy as np
from typing import Optional


def fix_box_feet_xml(input_xml_path: str, output_xml_path: Optional[str] = None) -> bool:
    """
    Create a corrected box feet XML that exactly matches LocoMujoco implementation
    """
    
    if output_xml_path is None:
        base_name = os.path.splitext(input_xml_path)[0]
        output_xml_path = f"{base_name}_corrected.xml"
    
    print(f"Fixing box feet implementation in {input_xml_path}")
    print(f"Output will be saved to {output_xml_path}")
    
    try:
        # Parse the XML
        tree = ET.parse(input_xml_path)
        root = tree.getroot()
        
        # LocoMujoco exact parameters
        scaling = 1.06552  # From XML scaling
        box_size_base = np.array([0.112, 0.03, 0.05])  # [length, height, width] - LocoMujoco values
        box_pos_base = np.array([-0.09, 0.019, 0.0])   # Position relative to toe body
        
        # Apply scaling (exactly as LocoMujoco does)
        box_size = box_size_base * scaling
        box_pos = box_pos_base * scaling
        
        print(f"LocoMujoco exact parameters:")
        print(f"  - Box size: [{box_size[0]:.4f}, {box_size[1]:.4f}, {box_size[2]:.4f}]")
        print(f"  - Box position: [{box_pos[0]:.4f}, {box_pos[1]:.4f}, {box_pos[2]:.4f}]")
        
        # Remove existing box feet if any
        boxes_removed = 0
        for geom in root.iter('geom'):
            if geom.get('name', '').startswith('foot_box'):
                parent = geom.getparent()
                if parent is not None:
                    parent.remove(geom)
                    boxes_removed += 1
        
        if boxes_removed > 0:
            print(f"  - Removed {boxes_removed} existing box geometries")
        
        # Step 1: Disable original foot collision geometries (LocoMujoco approach)
        original_foot_geoms = ["r_foot_col", "r_bofoot_col", "l_foot_col", "l_bofoot_col"]
        disabled_geoms = 0
        
        for geom in root.iter('geom'):
            geom_name = geom.get('name', '')
            if geom_name in original_foot_geoms:
                # LocoMujoco sets contype=0 and conaffinity=0 to disable collision
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
                disabled_geoms += 1
                print(f"  ✅ Disabled collision for {geom_name}")
        
        print(f"  - Disabled {disabled_geoms} original foot collision geometries")
        
        # Step 2: Add box feet geometries to toe bodies (exactly like LocoMujoco)
        toe_configs = [
            ("toes_l", [0.0, 0.15, 0.0]),   # Left foot: positive Y rotation
            ("toes_r", [0.0, -0.15, 0.0])   # Right foot: negative Y rotation
        ]
        
        boxes_added = 0
        
        # Find and modify toe bodies
        for body in root.iter('body'):
            body_name = body.get('name', '')
            
            for toe_name, euler_rotation in toe_configs:
                if body_name == toe_name:
                    print(f"  Found toe body: {toe_name}")
                    
                    # Create box geometry (matching LocoMujoco exactly)
                    box_geom = ET.SubElement(body, 'geom')
                    box_geom.set('name', f'foot_box_{toe_name}')
                    box_geom.set('type', 'box')
                    
                    # Size: exactly as LocoMujoco scales it
                    box_geom.set('size', f'{box_size[0]:.6f} {box_size[1]:.6f} {box_size[2]:.6f}')
                    
                    # Position: exactly as LocoMujoco positions it
                    box_geom.set('pos', f'{box_pos[0]:.6f} {box_pos[1]:.6f} {box_pos[2]:.6f}')
                    
                    # Rotation: exactly as LocoMujoco rotates it
                    box_geom.set('euler', f'{euler_rotation[0]} {euler_rotation[1]} {euler_rotation[2]}')
                    
                    # Visual: LocoMujoco uses semi-transparent gray
                    box_geom.set('rgba', '0.5 0.5 0.5 0.8')  # Gray with 80% opacity
                    
                    # Physics: Enable collision with ground
                    box_geom.set('condim', '3')      # 3D contact
                    box_geom.set('contype', '1')     # Collision type 1 (default)
                    box_geom.set('conaffinity', '1') # Collision affinity 1 (default)
                    
                    # Material properties for contact
                    box_geom.set('friction', '1.0 0.5 0.01')  # Static, sliding, rolling friction
                    box_geom.set('material', 'foot_material')   # Use foot material if available
                    
                    boxes_added += 1
                    print(f"    ✅ Added LocoMujoco-exact box geometry to {toe_name}")
                    print(f"       Size: {box_size}")
                    print(f"       Position: {box_pos}")
                    print(f"       Rotation: {euler_rotation}")
        
        # Step 3: Add material definition for foot boxes if not exists
        materials_section = root.find('.//default')
        if materials_section is not None:
            # Check if foot_material exists
            foot_material_exists = False
            for material in materials_section.iter('material'):
                if material.get('name') == 'foot_material':
                    foot_material_exists = True
                    break
            
            if not foot_material_exists:
                foot_material = ET.SubElement(materials_section, 'material')
                foot_material.set('name', 'foot_material')
                foot_material.set('rgba', '0.5 0.5 0.5 0.8')
                foot_material.set('specular', '0.3')
                foot_material.set('shininess', '0.1')
                print(f"    ✅ Added foot_material definition")
        
        # Save the corrected XML
        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
        
        print(f"\n✅ Box feet correction completed:")
        print(f"   - {disabled_geoms} original foot collision geometries disabled")
        print(f"   - {boxes_added} LocoMujoco-exact box geometries added")
        print(f"   - Both feet should now have visible gray box collision shapes")
        print(f"   - Output saved to: {output_xml_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix box feet: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fix box feet in the Genesis skeleton XML"""
    
    skeleton_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Start from the base XML (without box feet) for clean implementation
    input_xml = os.path.join(skeleton_dir, "revised_genesis_skeleton.xml")
    output_xml = os.path.join(skeleton_dir, "revised_genesis_skeleton_box_feet_corrected.xml")
    
    if not os.path.exists(input_xml):
        print(f"❌ Base XML file not found: {input_xml}")
        return
    
    print("🔧 Creating LocoMujoco-exact box feet implementation...")
    print("=" * 60)
    
    success = fix_box_feet_xml(input_xml, output_xml)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Box feet correction successful!")
        print("\nKey corrections made:")
        print("1. ✅ Disabled original foot mesh collision (contype=0, conaffinity=0)")
        print("2. ✅ Added exact LocoMujoco box dimensions and positioning")  
        print("3. ✅ Applied correct rotations (±0.15 rad Y-axis)")
        print("4. ✅ Set proper collision properties and materials")
        print("5. ✅ Both feet should now have visible gray box shapes")
        
        print(f"\n🎯 Next steps:")
        print(f"1. Update skeleton_humanoid.py to use: {os.path.basename(output_xml)}")
        print(f"2. Test in verify_trajectory.py - both feet should show gray boxes")
        print(f"3. Verify stable ground contact and trajectory following")
        
    else:
        print("\n❌ Box feet correction failed")


if __name__ == "__main__":
    main()