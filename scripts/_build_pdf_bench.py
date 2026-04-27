"""Generate 15 small PDF fixtures for the pdf benchmark and emit JSON items."""
import json
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from pypdf import PdfReader

FIX_DIR = "/Users/kevinzhang/Documents/GitHub/Toolflix/data/bench_fixtures/pdf"
OUT_PATH = "/Users/kevinzhang/Documents/GitHub/Toolflix/data/bench_v3_output/pdf.json"
REL_DIR = "data/bench_fixtures/pdf"

os.makedirs(FIX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

styles = getSampleStyleSheet()
title_style = styles["Title"]
h_style = styles["Heading2"]
body_style = styles["BodyText"]


def build_pdf(path, title, paragraphs, table=None):
    doc = SimpleDocTemplate(path, pagesize=LETTER, title=title)
    story = [Paragraph(title, title_style), Spacer(1, 0.2 * inch)]
    for p in paragraphs:
        story.append(Paragraph(p, body_style))
        story.append(Spacer(1, 0.12 * inch))
    if table is not None:
        t = Table(table, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(Spacer(1, 0.15 * inch))
        story.append(t)
    doc.build(story)


# Each entry: (filename, title, paragraphs, table_or_None, question, ground_truth)
SPECS = [
    # 01 - receipt: invoice total
    {
        "file": "sample_01.pdf",
        "title": "Acme Hardware Store - Receipt",
        "paragraphs": [
            "Transaction ID: AC-49217. Customer: Walter Monroe. Date of purchase: March 14, 2024.",
            "Payment processed via Visa ending in 4411. Store location: 82 Monroe Avenue, Trenton.",
        ],
        "table": [
            ["Item", "Qty", "Price"],
            ["Circular saw", "1", "$189.00"],
            ["Oak planks (10 ft)", "6", "$420.00"],
            ["Box of 3-inch screws", "3", "$36.00"],
            ["Subtotal", "", "$645.00"],
            ["Tax", "", "$51.60"],
            ["Grand Total", "", "$696.60"],
        ],
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_01.pdf. What is the grand total listed on the receipt? Return just the dollar amount.",
        "ground_truth": ["$696.60"],
    },
    # 02 - scientific abstract: accuracy number
    {
        "file": "sample_02.pdf",
        "title": "Abstract: Sparse Attention for Genomic Variant Calling",
        "paragraphs": [
            "Authors: Priya Raghavan, Marcus Holloway, Tomomi Saito. Affiliation: Helix Systems Lab.",
            "We introduce a sparse attention variant caller evaluated on the HG002 benchmark.",
            "Our method reaches an F1 score of 98.73 percent on single nucleotide variants, surpassing the prior sparse baseline by 2.4 points.",
            "Training used 384 A100-hours over the GRCh38 reference assembly.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_02.pdf. What F1 score does the abstract report on single nucleotide variants? Return just the percentage.",
        "ground_truth": ["98.73 percent"],
    },
    # 03 - recipe: flour quantity
    {
        "file": "sample_03.pdf",
        "title": "Grandma Elena's Sourdough Loaf",
        "paragraphs": [
            "Yields one large loaf. Preparation time: 30 minutes active, 14 hours resting.",
            "Ingredients: 612 grams of bread flour, 18 grams of fine sea salt, 120 grams of active starter, and 410 grams of filtered water.",
            "Bake in a covered Dutch oven at 475 F for 25 minutes, then uncover and bake 20 more minutes.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_03.pdf. How many grams of bread flour does the recipe call for? Return just the number with its unit.",
        "ground_truth": ["612 grams"],
    },
    # 04 - memo: signer name
    {
        "file": "sample_04.pdf",
        "title": "Internal Memorandum - Q4 Facilities Update",
        "paragraphs": [
            "To: All Building 7 staff. From: Facilities Office. Date: October 2, 2024.",
            "Effective November 1, the east wing loading dock will be closed for roof resurfacing.",
            "Please route all inbound shipments through the south dock during the 6-week closure.",
            "Signed: Harriet Calloway, Director of Facilities Operations.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_04.pdf. Who signed the memorandum? Return just the full name of the signer.",
        "ground_truth": ["Harriet Calloway"],
    },
    # 05 - specification: version
    {
        "file": "sample_05.pdf",
        "title": "Orbiter Navigation Module - Technical Specification",
        "paragraphs": [
            "Document owner: Guidance Systems Group. Revision released to manufacturing partners.",
            "Specification version: 7.4.2-beta. Prior stable release was 7.3.9 in February 2024.",
            "The module interfaces with the inertial measurement unit over a 1 MHz SPI bus.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_05.pdf. What specification version is listed for the navigation module? Return just the version string.",
        "ground_truth": ["7.4.2-beta"],
    },
    # 06 - sales table: highlighted cell
    {
        "file": "sample_06.pdf",
        "title": "Regional Sales Summary - FY2024",
        "paragraphs": [
            "The table below reports revenue by region in thousands of US dollars.",
        ],
        "table": [
            ["Region", "Q1", "Q2", "Q3", "Q4"],
            ["North", "412", "438", "501", "527"],
            ["South", "298", "311", "344", "362"],
            ["East", "556", "601", "649", "712"],
            ["West", "389", "402", "418", "455"],
        ],
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_06.pdf. What is the Q3 revenue figure for the East region in the sales table? Return just the number.",
        "ground_truth": ["649"],
    },
    # 07 - bibliography: year of a specific reference
    {
        "file": "sample_07.pdf",
        "title": "Selected Bibliography - History of Radio Astronomy",
        "paragraphs": [
            "[1] Jansky, K. G. Electrical disturbances of extraterrestrial origin. Proceedings of the IRE, 1933.",
            "[2] Reber, G. Cosmic static. Astrophysical Journal, 1944.",
            "[3] Ewen, H. I. and Purcell, E. M. Radiation from galactic hydrogen at 1420 megacycles. Nature, 1951.",
            "[4] Hewish, A. et al. Observation of a rapidly pulsating radio source. Nature, 1968.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_07.pdf. According to the bibliography, in what year was Grote Reber's 'Cosmic static' published? Return just the four-digit year.",
        "ground_truth": ["1944"],
    },
    # 08 - travel itinerary: flight number
    {
        "file": "sample_08.pdf",
        "title": "Travel Itinerary - San Francisco to Tokyo",
        "paragraphs": [
            "Traveler: Dr. Naomi Abe. Booking reference: XQ8-2201.",
            "Outbound flight: departs SFO at 11:45 AM on June 3, 2025.",
            "Flight number: UA837, operated by United Airlines, Boeing 787-9 aircraft.",
            "Arrives Narita at 3:20 PM local time on June 4.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_08.pdf. What is the outbound flight number listed on the itinerary? Return just the flight number.",
        "ground_truth": ["UA837"],
    },
    # 09 - contract: effective date
    {
        "file": "sample_09.pdf",
        "title": "Consulting Services Agreement",
        "paragraphs": [
            "This agreement is entered between Brightfield LLC (the Client) and Tomas Huang (the Consultant).",
            "Effective date: September 18, 2023. The initial term shall run for twelve months.",
            "Compensation is billed at 175 dollars per hour, invoiced monthly.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_09.pdf. What is the effective date of the consulting agreement? Return just the date.",
        "ground_truth": ["September 18, 2023"],
    },
    # 10 - lab report: measurement value
    {
        "file": "sample_10.pdf",
        "title": "Water Quality Lab Report - Sample WQ-1183",
        "paragraphs": [
            "Sample collected from the Clearwater Reservoir intake on August 7, 2024.",
            "Technician: Leila Osman. Instrument: Hach DR6000 spectrophotometer.",
            "Measured dissolved oxygen concentration: 8.42 mg/L. Measured pH: 7.31.",
            "All readings are within the acceptable municipal supply range.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_10.pdf. What dissolved oxygen concentration was measured for the sample? Return just the value with its unit.",
        "ground_truth": ["8.42 mg/L"],
    },
    # 11 - course syllabus: room number
    {
        "file": "sample_11.pdf",
        "title": "CS 318 - Distributed Systems - Spring 2025 Syllabus",
        "paragraphs": [
            "Instructor: Professor Amara Okonkwo. Teaching assistant: Rahul Venkatesh.",
            "Lectures meet Tuesdays and Thursdays from 10:30 to 11:45 AM in Kessler Hall, Room 214.",
            "Office hours are held Wednesdays from 2 to 4 PM in the third floor faculty suite.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_11.pdf. In which room do the lectures meet? Return just the building name and room number.",
        "ground_truth": ["Kessler Hall, Room 214"],
    },
    # 12 - warranty: coverage period
    {
        "file": "sample_12.pdf",
        "title": "Limited Warranty - Nordlys Espresso Machine Model E-220",
        "paragraphs": [
            "Nordlys Appliances warrants this product against defects in materials and workmanship.",
            "The coverage period for this warranty is 36 months from the date of purchase.",
            "Warranty claims require the original receipt and must be initiated through an authorized service center.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_12.pdf. How long is the coverage period for this limited warranty? Return just the duration.",
        "ground_truth": ["36 months"],
    },
    # 13 - conference schedule: keynote time
    {
        "file": "sample_13.pdf",
        "title": "Pacific Robotics Symposium - Day 2 Schedule",
        "paragraphs": [
            "The keynote address by Dr. Ingrid Sorensen begins at 9:15 AM in the Grand Ballroom.",
            "Poster session runs from 1:00 PM to 2:30 PM in the west concourse.",
            "Evening banquet is hosted at the Harborview Club starting at 7:00 PM.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_13.pdf. At what time does the keynote address begin on day 2? Return just the time of day.",
        "ground_truth": ["9:15 AM"],
    },
    # 14 - patient discharge summary: medication dose
    {
        "file": "sample_14.pdf",
        "title": "Patient Discharge Summary - MRN 773902",
        "paragraphs": [
            "Attending physician: Dr. Oluwaseun Adebayo. Discharge date: May 12, 2024.",
            "Home medication: patient is to take 250 mg of amoxicillin orally every 8 hours for 10 days.",
            "Follow-up appointment scheduled for May 26 at the Cedar Valley primary care clinic.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_14.pdf. What dose of amoxicillin is the patient instructed to take? Return just the dose with its unit.",
        "ground_truth": ["250 mg"],
    },
    # 15 - astronomy chart: distance value
    {
        "file": "sample_15.pdf",
        "title": "Exoplanet Data Sheet - Kepler-452b",
        "paragraphs": [
            "Discovery announced by the Kepler mission team in 2015.",
            "The host star is a G2-type main sequence star located in the constellation Cygnus.",
            "The system lies at an approximate distance of 1402 light-years from Earth.",
            "Orbital period of the planet: 384.8 Earth days.",
        ],
        "table": None,
        "question": "Read the PDF at data/bench_fixtures/pdf/sample_15.pdf. What approximate distance from Earth is given for the Kepler-452b system? Return just the distance with its unit.",
        "ground_truth": ["1402 light-years"],
    },
]


def extract_all_text(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


items = []
for idx, spec in enumerate(SPECS, start=1):
    path = os.path.join(FIX_DIR, spec["file"])
    build_pdf(path, spec["title"], spec["paragraphs"], spec["table"])
    text = extract_all_text(path)
    gt = spec["ground_truth"][0]
    # Verify by checking the key numeric/name token of gt appears in the extracted text.
    # We use a tolerant check: strip $, commas, and spaces.
    def normalize(s):
        return s.replace(",", "").replace(" ", "").lower()

    if normalize(gt) not in normalize(text):
        # Some tokens (like "$696.60") may survive; but "250 mg" may have whitespace differences.
        # Try a looser substring match on the raw text.
        if gt.lower() not in text.lower():
            raise SystemExit(
                f"Verification FAILED for {spec['file']}: ground truth '{gt}' not found in extracted text.\n---\n{text}\n---"
            )

    # Leakage check: gt must not appear as a substring of the question.
    if gt.lower() in spec["question"].lower():
        raise SystemExit(f"LEAKAGE in {spec['file']}: ground truth '{gt}' is substring of question.")

    rel_path = f"{REL_DIR}/{spec['file']}"
    items.append(
        {
            "id": f"pdf_{idx:03d}",
            "source": "constructed",
            "level": 1,
            "question": spec["question"],
            "expected_category": "pdf",
            "ground_truth": spec["ground_truth"],
            "attachments": [rel_path],
        }
    )
    print(f"OK  {spec['file']}  gt={gt!r}")

with open(OUT_PATH, "w") as f:
    json.dump(items, f, indent=2)

print(f"\nWrote {len(items)} items to {OUT_PATH}")
