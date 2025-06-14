import json
import re

def clean_description(text):
    """Remove noisy suffixes and truncate long descriptions."""
    text = text.replace("\n", " ").strip()
    text = re.split(r"---MISSING DATA|---SOURCE", text)[0].strip()
    return text[:300]

def bucket_priority(numeric_priority):
    """Convert numeric priority score to HIGH / MEDIUM / LOW."""
    try:
        score = int(numeric_priority)
        if score >= 900:
            return "HIGH"
        elif score >= 500:
            return "MEDIUM"
        else:
            return "LOW"
    except:
        return "LOW"

def generate_llm_attribute_reference(input_json_path, output_txt_path):
    with open(input_json_path) as f:
        attrs = json.load(f)

    lines = []
    for attr in sorted(attrs, key=lambda x: x.get("clinicalAttributeId", "")):
        attr_id = attr.get("clinicalAttributeId")
        if not attr_id:
            continue

        display_name = attr.get("displayName", "No display name").strip()
        description = clean_description(attr.get("description", "No description available"))
        dtype = attr.get("datatype", "unknown").lower()
        level = "patient-level" if attr.get("patientAttribute") else "sample-level"
        priority_bucket = bucket_priority(attr.get("priority", "0"))

        lines.append(
            f"- `{attr_id}` ({dtype}, {level}) | priority: {priority_bucket}\n"
            f"  Label: {display_name}\n"
            f"  Description: {description}"
        )

    with open(output_txt_path, "w") as out:
        out.write("\n\n".join(lines))

    print(f"✅ Attribute reference saved to {output_txt_path}")

# Example usage
if __name__ == "__main__":
    generate_llm_attribute_reference(
        input_json_path="attributes_description.json",
        output_txt_path="attribute_reference.txt"
    )
