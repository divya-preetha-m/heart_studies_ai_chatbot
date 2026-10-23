"""One-off script: generate sample_pdfs/sample_heart_anatomy.pdf (text-based,
2 pages) so the app can be tested without the student needing their own PDF.

Uses only pypdf (already a project dependency) — writes a minimal but valid PDF
with a content stream of Helvetica text lines.
"""

import os
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    ArrayObject,
)


def escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def page_content(lines: list[str]) -> bytes:
    """Build a PDF content stream: Helvetica 11pt, 16pt leading, top-down."""
    parts = ["BT", "/F1 11 Tf", "16 TL", "72 720 Td"]
    for line in lines:
        if line:
            parts.append(f"({escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def build_pdf(pages_lines: list[list[str]], out_path: str) -> None:
    writer = PdfWriter()
    for lines in pages_lines:
        page = writer.add_blank_page(width=612, height=792)

        # Font resource (Helvetica, built into every PDF reader)
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font)

        resources = DictionaryObject()
        resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
        page[NameObject("/Resources")] = resources

        # Content stream
        stream = DecodedStreamObject()
        stream.set_data(page_content(lines))
        page[NameObject("/Contents")] = writer._add_object(stream)

    with open(out_path, "wb") as fh:
        writer.write(fh)


PAGE_1 = [
    "HEART ANATOMY - STUDY NOTES (sample document)",
    "",
    "1. The four chambers",
    "The heart has four chambers. The right atrium receives oxygen-poor blood",
    "from the body through the superior and inferior vena cava. The right",
    "ventricle pumps that blood to the lungs through the pulmonary valve.",
    "The left atrium receives oxygen-rich blood from the lungs via the pulmonary",
    "veins. The left ventricle pumps oxygen-rich blood into the aorta; it has the",
    "thickest muscular wall because it must supply the entire body.",
    "",
    "2. The heart valves",
    "Four valves ensure one-way blood flow. The tricuspid valve lies between the",
    "right atrium and right ventricle. The mitral (bicuspid) valve lies between the",
    "left atrium and left ventricle. The pulmonary valve guards the exit of the",
    "right ventricle, and the aortic valve guards the exit of the left ventricle.",
    "The closing of these valves produces the classic 'lub-dub' heart sounds.",
    "",
    "3. Path of blood through the heart",
    "Body -> superior/inferior vena cava -> right atrium -> tricuspid valve ->",
    "right ventricle -> pulmonary valve -> pulmonary arteries -> lungs ->",
    "pulmonary veins -> left atrium -> mitral valve -> left ventricle ->",
    "aortic valve -> aorta -> body.",
]

PAGE_2 = [
    "HEART ANATOMY - STUDY NOTES (sample document), page 2",
    "",
    "4. Pulmonary and systemic circulation",
    "Pulmonary circulation carries blood between the heart and the lungs: the",
    "right ventricle pumps oxygen-poor blood to the lungs, and the pulmonary",
    "veins return oxygen-rich blood to the left atrium. Systemic circulation",
    "carries blood between the heart and the rest of the body: the left",
    "ventricle pumps oxygen-rich blood through the aorta, and the vena cavae",
    "return oxygen-poor blood to the right atrium. The two circuits operate in",
    "series, so blood passes through the heart once for every full body loop.",
    "",
    "5. The cardiac conduction system",
    "The heart generates its own electrical rhythm. The sinoatrial (SA) node in",
    "the right atrium is the natural pacemaker. It sends an impulse to the",
    "atrioventricular (AV) node, which delays the signal slightly so the atria",
    "can finish contracting. The impulse then travels down the Bundle of His,",
    "through the left and right bundle branches, and finally spreads through the",
    "Purkinje fibres, causing the ventricles to contract from the bottom up.",
    "",
    "6. Basic cardiovascular physiology",
    "Cardiac output is the volume of blood pumped per minute and equals heart",
    "rate multiplied by stroke volume. Blood pressure is roughly cardiac output",
    "times peripheral resistance. Arteries carry blood away from the heart,",
    "veins return blood to the heart, and capillaries are the thin vessels where",
    "oxygen, carbon dioxide and nutrients are exchanged with tissues.",
]


def main() -> None:
    os.makedirs("sample_pdfs", exist_ok=True)
    out = os.path.join("sample_pdfs", "sample_heart_anatomy.pdf")
    build_pdf([PAGE_1, PAGE_2], out)
    print(f"wrote {out}")

    # Verify pypdf can read it back (this is exactly what rag.py does)
    from pypdf import PdfReader

    reader = PdfReader(out)
    print(f"pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        print(f"page {i}: {len(text)} chars; first line: {text.splitlines()[0]!r}")


if __name__ == "__main__":
    main()
