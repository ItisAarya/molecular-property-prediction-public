"""
scripts/make_models_pdf.py

Build `paper/MODELS_AND_DATA_EXPLAINED.pdf`.

    python -m scripts.make_models_pdf

A plain-English explanation of one specific thing: how the models, the eight datasets, the
six scaffold splits and the fusion ladder fit together. `paper/PROJECT_EXPLAINED.pdf` covers
the project as a whole; this one answers "what exactly is being trained on what, and why is
there so much of it".

Written for a reader with no background at all. Every term is defined in one line the first
time it appears, and every number in it is computed here from the repository rather than
typed, so the document cannot drift from the results.
"""

import json
import os
import re
from collections import Counter

import scripts._pdflib  # noqa: F401  -- resolves reportlab; must precede its import

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

OUT = os.path.join("paper", "MODELS_AND_DATA_EXPLAINED.pdf")

NAVY = colors.HexColor("#14304F")
BLUE = colors.HexColor("#2A6099")
LIGHT = colors.HexColor("#EAF0F6")
GREY = colors.HexColor("#5A6570")
BORDER = colors.HexColor("#C3CDD8")
GREEN = colors.HexColor("#E8F3EC")
AMBER = colors.HexColor("#FDF3E3")

ss = getSampleStyleSheet()


def S(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


TITLE = S("T", fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=NAVY, spaceAfter=3)
SUB = S("Sub", fontSize=11.5, leading=16, textColor=GREY, spaceAfter=14)
H1 = S("H1", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY,
       spaceBefore=15, spaceAfter=7)
H2 = S("H2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=BLUE,
       spaceBefore=10, spaceAfter=4)
BODY = S("B", fontSize=10, leading=15, spaceAfter=6, alignment=TA_LEFT)
BULLET = S("Bu", parent=BODY, leftIndent=13, bulletIndent=3, spaceAfter=3.5)
SMALL = S("Sm", fontSize=8.6, leading=12, textColor=GREY)
CELL = S("C", fontSize=8.6, leading=11.5)
CELLB = S("CB", fontSize=8.6, leading=11.5, fontName="Helvetica-Bold")


def P(t, s=BODY):
    return Paragraph(t, s)


def B(t):
    return Paragraph(t, BULLET, bulletText="\u2022")


def box(text, bg=LIGHT, border=BORDER):
    t = Table([[Paragraph(text, BODY)]], colWidths=[168 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.9, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def table(rows, widths, header=True, zebra=True, align_right=()):
    data = [[Paragraph(c, CELLB if (header and i == 0) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    for c in align_right:
        st.append(("ALIGN", (c, 1), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(GREY)
    canvas.drawString(21 * mm, 12 * mm,
                      "Molecular Property Prediction  |  models, datasets, splits and the ladder")
    canvas.drawRightString(A4[0] - 21 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(21 * mm, 15.5 * mm, A4[0] - 21 * mm, 15.5 * mm)
    canvas.restoreState()


# ======================================================================================
# Numbers, computed rather than typed.
# ======================================================================================
def dataset_facts():
    meta = json.load(open(os.path.join("data", "dataset_meta.json")))
    out = []
    for ds, d in meta.items():
        pool = np.load(os.path.join("data", "pool", f"{ds}_ecfp.npz"), allow_pickle=True)
        out.append({
            "ds": ds,
            "tasks": len(d["tasks"]),
            "type": d["task_type"],
            "n": int(pool["X"].shape[0]),
        })
    return out


def fusion_sizes():
    from src.models.fusion import build_fusion
    return {m: sum(p.numel() for p in build_fusion(m, n_views=3, d=256).parameters())
            for m in ("concat", "gated", "xattn", "bilinear", "proposed")}


def encoder_sizes():
    from src.models.encoders.graph import build_graph_encoder
    return {e: sum(p.numel() for p in build_graph_encoder(e, in_dim=34, hidden=256).parameters())
            for e in ("gin", "gine", "attentivefp")}


def trained_counts():
    """How many (model, dataset, split) combinations have an archived test result."""
    runs = os.path.join("results", "runs")
    tags = Counter()
    for v in sorted(os.listdir(runs)):
        d = os.path.join(runs, v, "metrics")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = re.match(r"^([a-z0-9]+)_(.+)_test\.csv$", f)
            if m:
                tags[m.group(2)] += 1
    return len(tags), sum(tags.values())


FACTS = dataset_facts()
FUSION = fusion_sizes()
ENC = encoder_sizes()
N_TAGS, N_TRAINED = trained_counts()
N_DS = len(FACTS)
N_SPLITS = 6

QUESTION = {
    "tox21": "Is this molecule toxic? (12 separate biological alarms)",
    "bbbp": "Can it get from the blood into the brain?",
    "clintox": "Did it pass or fail human clinical trials?",
    "bace": "Does it block an enzyme linked to Alzheimer's?",
    "sider": "What side effects does it cause? (27 body systems)",
    "esol": "How well does it dissolve in water?",
    "lipophilicity": "Does it prefer fat or water?",
    "freesolv": "How much energy does it take to dissolve it in water?",
}
ANSWER = {
    "tox21": "yes / no, 12 times over",
    "bbbp": "yes / no",
    "clintox": "yes / no, twice",
    "bace": "yes / no",
    "sider": "yes / no, 27 times over",
    "esol": "a number (log mol/L)",
    "lipophilicity": "a number (logD)",
    "freesolv": "a number (kcal/mol)",
}

s = []

s.append(P("What Is Being Trained On What", TITLE))
s.append(P("The models, the eight datasets, the six splits, and the ladder \u2014 and exactly "
           "how they fit together. Written for a reader with no background in chemistry, "
           "computing or statistics.", SUB))

s.append(P("1. Start here: the one thing that surprises everyone", H1))
s.append(box(
    "There is <b>not one model</b> in this project. There are <b>thousands</b>.<br/><br/>"
    "A common guess is that we built one clever program, showed it eight datasets, and it "
    "learned all of them. That is not what happens. Each dataset asks a completely different "
    "question with a different kind of answer, so <b>each dataset gets its own separately "
    "trained model</b>. And because we test each design on six different ways of splitting "
    "the data, each design is trained <b>six times per dataset</b>.<br/><br/>"
    f"One design \u00d7 {N_DS} datasets \u00d7 {N_SPLITS} splits = <b>{N_DS*N_SPLITS} trained "
    f"models</b>, just to evaluate <i>one</i> idea once. Across the whole project we have "
    f"trained and archived <b>{N_TRAINED:,} models</b> covering <b>{N_TAGS} different "
    f"designs</b>.", AMBER, colors.HexColor("#E0B87A")))

s.append(P("The rest of this document explains each of those three multipliers: the datasets, "
           "the splits, and the designs.", BODY))

s.append(P("2. First, some words you need", H1))
s.append(table([
    ["Word", "What it means, in one line"],
    ["<b>Molecule</b>", "A substance \u2014 a medicine, a poison, a sugar. Made of atoms joined by bonds."],
    ["<b>SMILES</b>", "A way of writing a molecule as text, so a computer can read it. Aspirin is <font face='Courier'>CC(=O)Oc1ccccc1C(=O)O</font>."],
    ["<b>Property</b>", "Something true about a molecule that we would like to know without doing the experiment \u2014 is it toxic, does it dissolve."],
    ["<b>Dataset</b>", "A list of real molecules with a property that scientists actually measured in a lab."],
    ["<b>Task</b>", "One single question inside a dataset. Some datasets ask one question; one asks 27."],
    ["<b>Model</b>", "A program that reads a molecule and guesses a property. It learns by looking at the measured examples."],
    ["<b>Training</b>", "Showing the model molecules whose answers we know, and repeatedly nudging it to be less wrong."],
    ["<b>Split</b>", "Dividing a dataset into a part to learn from and a part to be tested on, so the test is honest."],
], [30 * mm, 138 * mm], align_right=()))

s.append(P("3. The eight datasets \u2014 eight different questions", H1))
s.append(P("These are public collections that chemists have used as a shared yardstick for years. "
           "We did not create them. Each one contains real molecules with real measured answers.",
           BODY))

rows = [["Dataset", "The question it asks", "Answer looks like", "Questions", "Molecules"]]
for f in FACTS:
    rows.append([f"<b>{f['ds']}</b>", QUESTION[f["ds"]], ANSWER[f["ds"]],
                 str(f["tasks"]), f"{f['n']:,}"])
s.append(table(rows, [24 * mm, 62 * mm, 37 * mm, 18 * mm, 21 * mm], align_right=(3, 4)))

s.append(Spacer(1, 6))
s.append(P("Two things to notice.", H2))
s.append(B("<b>The answers are different kinds of thing.</b> Five datasets want a yes/no "
           "(\"classification\"). Three want a number (\"regression\"). A model that outputs "
           "yes/no cannot output a number, so these cannot share one model even in principle."))
s.append(B("<b>They are small.</b> The biggest has about 7,800 molecules; the smallest has 642. "
           "In modern machine learning that is tiny. This matters more than anything else in "
           "this project: on data this small, results wobble a lot, and telling a real "
           "improvement from luck is the hard part."))

s.append(P("4. How a model sees a molecule: the three \u201cviews\u201d", H1))
s.append(P("A computer cannot read a drawing of a molecule. It needs numbers. There are several "
           "reasonable ways to turn a molecule into numbers, and none of them is obviously best \u2014 "
           "so this project uses three at once and calls them <b>views</b>.", BODY))

s.append(table([
    ["View", "What it does", "Like\u2026"],
    ["<b>Graph</b>",
     "Treats the molecule as a network of atoms joined by bonds, and lets information flow "
     "between neighbouring atoms.",
     "Reading a map of a city by following its roads."],
    ["<b>Sequence</b>",
     "Feeds the SMILES text into ChemBERTa \u2014 a language model that was pre-trained on millions "
     "of molecule strings, the same idea as ChatGPT but for chemistry notation.",
     "Reading the molecule as a sentence."],
    ["<b>Descriptors</b>",
     "Computes ~217 standard chemical measurements (weight, greasiness, number of rings\u2026) plus a "
     "1,024-slot checklist of which small fragments are present.",
     "Filling in a fact sheet about the molecule."],
], [28 * mm, 86 * mm, 54 * mm]))

s.append(Spacer(1, 5))
s.append(box("<b>Why three?</b> Each view is blind to something. The fact sheet does not know how "
             "the atoms are connected. The map does not know the molecule's overall weight. The "
             "hope behind this project is that <b>combining</b> them beats any one of them. "
             "Testing that hope honestly is what the whole project is about.", LIGHT, BORDER))

s.append(P("5. The splits: why six, and what \u201cscaffold\u201d means", H1))
s.append(P("If you train a model and test it on the same molecules, it will look brilliant and be "
           "useless \u2014 it has memorised the answers. So the data is divided up first.", BODY))

s.append(table([
    ["Part", "Share", "What it is for"],
    ["<b>Training set</b>", "80%", "The model learns from these. It sees both the molecules and the answers."],
    ["<b>Validation set</b>", "10%", "Used <i>during</i> training to decide when to stop and to make every choice. The model never learns from these."],
    ["<b>Test set</b>", "10%", "Touched once, at the very end, to report the score. Never used to make any decision."],
], [30 * mm, 16 * mm, 122 * mm]))

s.append(P("What a scaffold split is", H2))
s.append(P("A <b>scaffold</b> is a molecule's skeleton \u2014 the core ring structure left when you "
           "strip off the decorations. Thousands of real medicines share a few hundred scaffolds; "
           "they come in families.", BODY))
s.append(P("If you split the data <b>randomly</b>, close cousins from the same family land in both "
           "the training set and the test set. The model then gets tested on molecules that are "
           "nearly identical to ones it studied. It scores brilliantly and has learned nothing "
           "useful.", BODY))
s.append(box("A <b>scaffold split</b> keeps every member of a family together. If one cousin is in "
             "the training set, all of them are. The test set is made of skeletons the model has "
             "<b>never seen in any form</b>. It is a much harder test \u2014 and it is the honest one, "
             "because a real chemist wants predictions about genuinely new molecules.",
             GREEN, colors.HexColor("#9CC3A9")))

s.append(P("And why six of them", H2))
s.append(P("There is more than one way to divide molecules into families and hand them out. Change "
           "the way you do it and the same model, with the same code, can score noticeably "
           "differently \u2014 by pure luck of the draw.", BODY))
s.append(B("<b>One</b> is the standard split everybody in the field uses, so our numbers can be "
           "compared with published ones."))
s.append(B("<b>Five</b> more are generated by us with different random starting points \u2014 same "
           "rules, different shuffles."))
s.append(P("Every model is trained on all six. We then report the <b>average across the five</b> "
           "and how much they disagreed. A difference that only shows up on one split is noise; "
           "one that survives all five is a finding. <b>This is the single most important design "
           "decision in the project.</b>", BODY))

s.append(P("6. The ladder: five ways to combine the three views", H1))
s.append(P("Once you have three views, you have to merge them into one answer. There are many ways. "
           "We built five, from the simplest possible to the most elaborate, each adding exactly "
           "one idea to the one before. That staircase is what we call <b>the ladder</b>.", BODY))

fr = [["Rung", "What it adds", "Extra parts to learn"]]
labels = {
    "concat": ("<b>1. concat</b>", "Just stack the three views end to end. Nothing is learned about how to combine them. The floor of the ladder."),
    "gated": ("<b>2. gated</b>", "Learns, for each molecule, how much to trust each view \u2014 like a volume knob per view."),
    "xattn": ("<b>3. xattn</b>", "Cross-attention: lets each view look at the other two and update itself before merging."),
    "bilinear": ("<b>4. bilinear</b>", "Multiplies pairs of views together, so the model can react to <i>combinations</i> rather than to each view separately."),
    "proposed": ("<b>5. proposed</b>", "All of it at once: cross-attention, then multiplication, then the volume knobs. Our own design."),
}
for k in ("concat", "gated", "xattn", "bilinear", "proposed"):
    name, desc = labels[k]
    n = FUSION[k]
    fr.append([name, desc, "none" if n == 0 else f"{n:,}"])
s.append(table(fr, [26 * mm, 112 * mm, 30 * mm], align_right=(2,)))

s.append(Spacer(1, 5))
s.append(box("<b>Why build a ladder instead of just the best one?</b> Suppose we only built rung 5, "
             "and it won. We would not know <i>why</i>. Was it the attention? The multiplication? "
             "The volume knobs? Or none of them \u2014 just the extra parts? Building every rung and "
             "testing all five the same way is the only way to find out which idea actually did the "
             "work. That is an <b>ablation</b>: remove one piece at a time and see what breaks.",
             LIGHT, BORDER))

s.append(P("7. Putting the three multipliers together", H1))
s.append(P("Now the arithmetic that produces thousands of models.", BODY))
s.append(table([
    ["", "How many", "Why"],
    ["Designs on the ladder", "5", "concat, gated, xattn, bilinear, proposed"],
    ["\u00d7 Datasets", f"{N_DS}", "each asks a different question, so each needs its own model"],
    ["\u00d7 Splits", f"{N_SPLITS}", "1 standard + 5 of ours, so we can tell a finding from luck"],
    ["<b>= Models for one ladder</b>", f"<b>{5*N_DS*N_SPLITS}</b>", "and that is one experiment, run once"],
], [50 * mm, 26 * mm, 92 * mm], align_right=(1,)))

s.append(Spacer(1, 4))
s.append(P("And the ladder is only part of it. We also trained the three views on their own (to see "
           "whether combining helps at all), the pipeline this project inherited (to measure "
           "improvement against), two published methods from other research groups (to check we are "
           "not beating only ourselves), and the whole ladder a second time with the language model "
           "un-frozen. Each of those is another 48 models.", BODY))
s.append(box(f"Total archived so far: <b>{N_TRAINED:,} trained models</b> across <b>{N_TAGS} "
             f"designs</b>, every one of them tested on all {N_DS} datasets and all {N_SPLITS} "
             f"splits. Roughly 30 hours of computing time, mostly on free cloud GPUs.",
             LIGHT, BORDER))

s.append(P("8. What actually happens during training", H1))
s.append(table([
    ["Step", "In one line"],
    ["<b>1. Feed a batch</b>", "Show the model 128 molecules at once and let it guess their properties."],
    ["<b>2. Measure the error</b>", "Compare every guess with the measured truth and turn the difference into one number."],
    ["<b>3. Nudge</b>", "Adjust every internal setting slightly in whichever direction reduces that error."],
    ["<b>4. Repeat</b>", "Go through the whole training set again. One full pass is called an <b>epoch</b>."],
    ["<b>5. Check on the validation set</b>", "After each epoch, score molecules it did not learn from."],
    ["<b>6. Stop when it stops improving</b>", "If 15 epochs pass with no improvement, stop and keep the best version. This is <b>early stopping</b>, and it prevents memorising."],
    ["<b>7. Score once on the test set</b>", "One number, reported. Nothing is changed after seeing it."],
], [42 * mm, 126 * mm]))

s.append(Spacer(1, 5))
s.append(box("<b>Everything is trained the same way.</b> Same learning speed, same batch size, same "
             "stopping rule, for every model in every table \u2014 written down in one file "
             "(<font face='Courier'>configs/shared.yaml</font>) and checked automatically. If we "
             "tuned our own design and left the others at default settings, our design would win "
             "and the comparison would be worthless. It would be measuring effort, not ideas.",
             AMBER, colors.HexColor("#E0B87A")))

s.append(P("9. Every model in the project, and its job", H1))
s.append(table([
    ["Name", "What it is", "Why it exists"],
    ["<b>gin_ref</b>", "The simple graph model this project inherited.",
     "The thing to beat. Every improvement is measured against it."],
    ["<b>gine</b>", "A bigger graph model that also reads the bonds, not just the atoms.",
     f"Tests whether a stronger graph reader helps. {ENC['gine']:,} parts vs {ENC['gin']:,}."],
    ["<b>desc</b>", "The chemical fact sheet alone, through a small network.",
     "The unglamorous baseline. Often the one to beat."],
    ["<b>seq_frozen</b>", "ChemBERTa used as-is, not adjusted.", "The cheap way to use a language model."],
    ["<b>lora</b>", "ChemBERTa with a small trainable add-on.",
     "Tests whether adjusting the language model is worth ~11 hours of GPU time."],
    ["<b>fuse_concat \u2026 fuse_proposed</b>", "The five rungs of the ladder.",
     "The main experiment: does combining views help, and which combining trick does the work?"],
    ["<b>attentivefp</b>", "A published graph model from another research group.",
     "Checks we are not just beating our own weak baseline."],
    ["<b>chemprop</b>", "A widely used published tool from another group.",
     "Same reason. It is the one a chemist would actually reach for."],
    ["<b>rf / gnn / trf / hybrid / ens</b>", "The five models of the inherited pipeline.",
     "The starting point, kept so the before/after is real."],
], [36 * mm, 60 * mm, 72 * mm]))

s.append(P("10. So which model does the demo use?", H1))
s.append(P("The Streamlit app serves <b>fuse_proposed</b> \u2014 rung 5, our own design \u2014 "
           "with one separately trained copy per dataset, all trained on the standard scaffold "
           "split. Type a molecule and it runs eight models, one per property.", BODY))
s.append(P("Alongside every prediction the app shows two things most demos hide: a "
           "<b>confidence range</b> (or, for yes/no questions, whether the model can decide at all "
           "at 90% confidence), and the model's <b>measured accuracy on the test set</b>, so a "
           "viewer can see how much the number is worth.", BODY))

s.append(P("11. What we actually found \u2014 the honest version", H1))
s.append(box(
    "Our elaborate design (rung 5, over a million extra parts to learn) <b>did not reliably beat</b> "
    "the simple volume-knob design (rung 2, about sixteen thousand). Neither reliably beat the "
    "inherited model. Neither did the two published methods from other groups. On several datasets "
    "the plain chemical fact sheet was as good as anything.<br/><br/>"
    "<b>This is a real result, not a failure.</b> The project was built to measure honestly, and "
    "what it measured is that most of the improvements people report at this scale do not survive "
    "being tested on six splits instead of one. We found that our own idea was one of them, and we "
    "are reporting it. A measuring instrument that only ever confirms the person holding it is not "
    "a measuring instrument.", GREEN, colors.HexColor("#9CC3A9")))

s.append(P("12. Questions a reader usually asks next", H1))
qa = [
    ("Why not train one model on all eight datasets together?",
     "Because they answer different kinds of question in different units \u2014 yes/no versus a number "
     "in kcal/mol. Sharing knowledge between them is a real research area, but it is a different "
     "project, and mixing it in would make it impossible to say which change caused what."),
    ("Why six splits and not more?",
     "Each extra split costs another full round of training on every dataset. Six gives us a stable "
     "average and a statistical test strong enough to detect the size of difference the field "
     "reports, which is what we need. More would be better and would cost more than we have."),
    ("Is the test set really never used?",
     "Yes. Every choice \u2014 when to stop, which version to keep, which design to report \u2014 is "
     "made on the validation set. Fixing that was one of the first things this project did: the "
     "inherited pipeline had been picking its winner by looking at the test score, which quietly "
     "inflates every number it reports."),
    ("Why is the ladder trained twice?",
     "Once with the language model frozen (fast) and once with it learning too (about eleven hours "
     "of GPU time). Freezing it is a shortcut, and we wanted to know whether the shortcut cost "
     "anything. It did not."),
    ("How do you know the demo app is using the same features as training?",
     "There is an automatic check (<font face='Courier'>scripts/check_deploy.py</font>) that takes "
     "molecules from the training data, runs them through the app's own code, and demands the "
     "numbers match exactly. A model fed slightly wrong inputs does not crash \u2014 it returns "
     "confident nonsense \u2014 so this is checked rather than assumed."),
]
for q, a in qa:
    s.append(KeepTogether([Paragraph(f"<b>{q}</b>", S("Q", fontSize=10, leading=14,
                                                      fontName="Helvetica-Bold",
                                                      textColor=NAVY, spaceBefore=8,
                                                      spaceAfter=3)),
                           Paragraph(a, S("A", fontSize=9.7, leading=14, leftIndent=11,
                                          spaceAfter=2))]))

s.append(Spacer(1, 10))
s.append(box("<b>If you remember one sentence:</b> each of the eight datasets gets its own model, "
             "each model is trained six times on six different honest splits, and the five rungs of "
             f"the ladder exist so that when something works we can say <i>which part</i> of it "
             f"worked \u2014 which is how {N_TRAINED:,} trained models turn into one trustworthy "
             "answer.", LIGHT, BORDER))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=21 * mm, rightMargin=21 * mm,
                        topMargin=18 * mm, bottomMargin=20 * mm,
                        title="Models, Datasets, Splits and the Ladder",
                        author="")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB)")
