"""
Copyright © Rowan Ramamurthy, 2025.
This software is provided "as-is," without any express or implied warranty. 
In no event shall the authors be held liable for any damages arising from 
the use of this software.

Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it 
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Author: Rowan Ramamurthy
"""
import pandas as pd

def parse_location(location: str):
    """Extract state from US locations"""
    if pd.isna(location):
        return None, None
    parts = [p.strip().lower() for p in location.split(",")]
    if "usa" in parts[-1].lower():
        if len(parts) >= 3:
            return parts[-1].strip(), parts[-2].strip().replace("'", "")
        elif len(parts) == 2:
            return parts[-1].strip(), parts[0].strip()
    return None, None

def create_accent_id(row, accent_map):
    """Generate consistent accent identifier"""
    country = str(row['country']).lower().replace(" ", "_")
    gender = str(row['sex']).lower()
    
    # Special handling for US states
    if country == "usa":
        state = None
        if not pd.isna(row['birthplace']):
            _, state = parse_location(row['birthplace'])
        state = state or "unknown"
        accent_id = f"{gender}_usa_{state}"
    else:
        accent_id = f"{gender}_{country}"
    
    # Add to map if new
    if accent_id not in accent_map:
        accent_map[accent_id] = len(accent_map)
    return accent_id