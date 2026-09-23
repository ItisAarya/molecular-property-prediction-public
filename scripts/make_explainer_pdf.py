import os
import scripts._pdflib  # noqa: F401  -- resolves reportlab; must precede its import

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, KeepTogether)
from reportlab.lib.enums import TA_LEFT

OUT = os.path.join("paper", "PROJECT_EXPLAINED.pdf")

NAVY   = colors.HexColor("#14304F")
BLUE   = colors.HexColor("#2A6099")
LIGHT  = colors.HexColor("#EAF0F6")
GREY   = colors.HexColor("#5A6570")
BORDER = colors.HexColor("#C3CDD8")
GREEN  = colors.HexColor("#E8F3EC")
AMBER  = colors.HexColor("#FDF3E3")

ss = getSampleStyleSheet()
def S(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

TITLE   = S("T", fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=NAVY, spaceAfter=3)
SUB     = S("Sub", fontSize=11.5, leading=16, textColor=GREY, spaceAfter=14)
H1      = S("H1", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY,
            spaceBefore=15, spaceAfter=7)
H2      = S("H2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=BLUE,
            spaceBefore=10, spaceAfter=4)
BODY    = S("B", fontSize=10, leading=15, spaceAfter=6, alignment=TA_LEFT)
BULLET  = S("Bu", parent=BODY, leftIndent=13, bulletIndent=3, spaceAfter=3.5)
SMALL   = S("Sm", fontSize=8.6, leading=12, textColor=GREY)
CELL    = S("C", fontSize=8.8, leading=12)
CELLB   = S("CB", fontSize=8.8, leading=12, fontName="Helvetica-Bold")
QSTYLE  = S("Q", fontSize=10, leading=14, fontName="Helvetica-Bold", textColor=NAVY,
            spaceBefore=8, spaceAfter=3)
ASTYLE  = S("A", fontSize=9.7, leading=14, leftIndent=11, spaceAfter=2)

def P(t, s=BODY):  return Paragraph(t, s)
def B(t):          return Paragraph(t, BULLET, bulletText="•")

def box(text, bg, border):
    t = Table([[Paragraph(text, BODY)]], colWidths=[168*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg), ("BOX", (0,0), (-1,-1), 0.9, border),
        ("LEFTPADDING",(0,0),(-1,-1),9), ("RIGHTPADDING",(0,0),(-1,-1),9),
        ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    return t

def table(rows, widths, header=True, zebra=True):
    data = [[Paragraph(c, CELLB if (header and i == 0) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN",(0,0),(-1,-1),"TOP"),
          ("GRID",(0,0),(-1,-1),0.4,BORDER),
          ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
          ("TOPPADDING",(0,0),(-1,-1),4.5),("BOTTOMPADDING",(0,0),(-1,-1),4.5)]
    if header:
        st += [("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white)]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0: st.append(("BACKGROUND",(0,i),(-1,i),LIGHT))
    t.setStyle(TableStyle(st))
    return t

def flow(steps):
    """A simple left-to-right arrow chain."""
    cells, w = [], []
    for i, s in enumerate(steps):
        cells.append(Paragraph(s, ParagraphStyle("f", parent=CELL, alignment=1,
                                                 fontName="Helvetica-Bold", fontSize=8.4)))
        w.append(30*mm)
        if i < len(steps)-1:
            cells.append(Paragraph("&rarr;", ParagraphStyle("a", parent=CELL, alignment=1,
                                                            fontSize=13, textColor=BLUE)))
            w.append(7*mm)
    t = Table([cells], colWidths=w)
    t.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BACKGROUND",(0,0),(0,0),LIGHT),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)] +
        [("BOX",(i,0),(i,0),0.8,BORDER) for i in range(0,len(cells),2)] +
        [("BACKGROUND",(i,0),(i,0),LIGHT) for i in range(0,len(cells),2)]))
    return t

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6); canvas.setFillColor(GREY)
    canvas.drawString(21*mm, 12*mm, "Molecular Property Prediction  |  project explainer")
    canvas.drawRightString(A4[0]-21*mm, 12*mm, f"Page {doc.page}")
    canvas.setStrokeColor(BORDER); canvas.setLineWidth(0.4)
    canvas.line(21*mm, 15.5*mm, A4[0]-21*mm, 15.5*mm)
    canvas.restoreState()

s = []

# ───────────────────────── PAGE 1 ─────────────────────────
s.append(P("Predicting What a Medicine Will Do, Before Making It", TITLE))
s.append(P("A plain-English guide to this project: what it does, how it works, "
           "what it is built with, and what we found. No science background needed.", SUB))

s.append(P("1. The one-paragraph version", H1))
s.append(box(
    "Making a new medicine is slow and expensive. Before a chemist makes a substance in the lab, "
    "it helps to guess whether it will be useful or harmful. This project builds computer programs "
    "that look at the <b>recipe of a molecule</b> and predict things like \"is this toxic?\" or "
    "\"will this dissolve in water?\". <b>But the real point of the project is not the predictor. "
    "It is the measuring.</b> We built a very strict way of testing whether such programs actually "
    "work, and then used it on our own program. It turned out that our fancy program was no better "
    "than a simple one, and part of its apparent skill came from a hidden clue in the data. We are reporting that honestly, because most papers in this field never "
    "check carefully enough to find out.", LIGHT, BORDER))

s.append(P("2. Key words, one line each", H1))
s.append(P("Everything in this project is built out of these ideas.", BODY))
s.append(table([
 ["Word", "What it means in one line"],
 ["Molecule", "A substance, like caffeine or aspirin, made of atoms joined together."],
 ["Property", "Something we want to know about it, like \"is it poisonous?\" or \"does it dissolve?\"."],
 ["Model", "A computer program that learns patterns from examples and then makes guesses."],
 ["Training", "Showing the program thousands of known examples so it can learn the pattern."],
 ["Dataset", "A big table of molecules where the answer is already known, used for training and testing."],
 ["Prediction", "The program's guess for a molecule it has never seen before."],
 ["Feature / representation", "A way of turning a molecule into numbers, because computers only handle numbers."],
 ["Baseline", "A simple method we compare against, to check whether the complicated one is worth it."],
 ["AUC", "A score from 0.5 to 1.0 for yes/no questions. 0.5 is random guessing, 1.0 is perfect."],
 ["RMSE", "An error score for number questions. Lower is better. 0 would be perfect."],
], [40*mm, 128*mm]))

s.append(P("3. What problem are we solving?", H1))
s.append(P("A chemist can draw a molecule on paper long before making it in the lab. Making it "
           "costs money and months of work. So the question is:", BODY))
s.append(box("<b>Given only the drawing of a molecule, can a computer predict what it will do?</b>",
             GREEN, colors.HexColor("#9CC3A9")))
s.append(P("If yes, chemists can test thousands of ideas on a laptop overnight and only make the "
           "most promising few in the lab. That is why this field exists.", BODY))

s.append(PageBreak())

# ───────────────────────── PAGE 2 ─────────────────────────
s.append(P("4. The data we use", H1))
s.append(P("We use <b>8 public datasets</b> called MoleculeNet. They are the standard test sets "
           "in this field, so our results can be compared with other people's. Each one is a table: "
           "one row per molecule, with the correct answer already filled in by real lab experiments.", BODY))
s.append(table([
 ["Dataset", "How many molecules", "What it asks", "Type"],
 ["Tox21", "7,823", "Is this toxic? (12 different toxicity tests)", "Yes / No"],
 ["BBBP", "2,039", "Can this reach the brain through the blood?", "Yes / No"],
 ["ClinTox", "1,480", "Did this fail safety trials in humans?", "Yes / No"],
 ["BACE", "1,513", "Does this block an enzyme linked to Alzheimer's?", "Yes / No"],
 ["SIDER", "1,427", "What side effects does this drug have? (27 kinds)", "Yes / No"],
 ["ESOL", "1,128", "How well does this dissolve in water?", "A number"],
 ["Lipophilicity", "4,200", "Does this prefer fat or water?", "A number"],
 ["FreeSolv", "642", "How much energy to dissolve it in water?", "A number"],
], [28*mm, 27*mm, 79*mm, 22*mm]))
s.append(P("Five datasets ask <b>yes/no</b> questions (scored with AUC) and three ask for "
           "<b>a number</b> (scored with RMSE). We deliberately left out a ninth dataset, HIV, "
           "because it is huge and would have cost about 60 hours of computing without making our "
           "conclusions any stronger.", BODY))

s.append(P("5. How a molecule becomes numbers", H1))
s.append(P("A computer cannot read a drawing of a molecule. So we describe the same molecule in "
           "<b>five different ways</b>. Each way notices something the others miss. We call each one "
           "a <b>view</b>, like photographing the same object from five angles.", BODY))
s.append(table([
 ["View", "How it describes the molecule", "What it is good at"],
 ["Graph (GIN)", "As a network: atoms are dots, bonds are lines joining them.",
  "The shape and how atoms connect."],
 ["Graph (GINE)", "The same, but it also notices what <i>kind</i> of bond each line is.",
  "A little more detail than GIN."],
 ["AttentiveFP", "Also a network, but it learns which atoms to pay most attention to.",
  "A well-known published method; we use it as a fair comparison."],
 ["Sequence (ChemBERTa)", "As text. Chemists write molecules as strings like \"CCO\" for alcohol.",
  "Patterns in how chemists write molecules."],
 ["Descriptor / fingerprint", "As a long list of measured facts: weight, size, oiliness, and which "
  "small fragments appear.", "Known chemistry, calculated directly."],
], [33*mm, 76*mm, 59*mm]))
s.append(P("<b>ChemBERTa</b> is worth one extra line: it is a language model, like a tiny cousin of "
           "ChatGPT, but it was trained on millions of molecule strings instead of English sentences. "
           "It already knows a lot about chemistry before we start.", SMALL))

s.append(PageBreak())

# ───────────────────────── PAGE 3 ─────────────────────────
s.append(P("6. The architecture (how the program is built)", H1))
s.append(P("The whole system is a pipeline. A molecule goes in one end and a prediction comes out "
           "the other.", BODY))
s.append(Spacer(1, 4))
s.append(flow(["Molecule", "Three views", "Make same size", "Fusion", "Head", "Prediction"]))
s.append(Spacer(1, 8))
s.append(table([
 ["Stage", "What happens here, in plain words"],
 ["1. Molecule", "One substance, written as a text string by a chemist."],
 ["2. Three views", "The combined model describes it three ways at once: a network (GINE), text read "
  "by ChemBERTa, and a list of chemical facts. Each view produces its own list of numbers."],
 ["3. Make same size", "The three lists come out different lengths, so we shrink or stretch each one "
  "to exactly 256 numbers. This is important for fairness, see below."],
 ["4. Fusion", "The step that <b>combines</b> the three descriptions into one. This was meant to be "
  "the clever part of the project."],
 ["5. Head", "A small final calculator that turns the combined description into the actual answer."],
 ["6. Prediction", "\"87% likely to be toxic\", or \"dissolves at -3.2\"."],
], [30*mm, 138*mm]))

s.append(P("Why step 3 matters (a mistake we caught)", H2))
s.append(box(
 "At first, each view fed a final calculator sized to match it. The text view produced a big list, "
 "so it got a big calculator; others got small ones. That meant when one view won, we could not tell "
 "if it was because the <b>description</b> was better or just because its <b>calculator was bigger</b>. "
 "We fixed this by forcing every view to exactly 256 numbers and giving every view the same size "
 "calculator. Then we re-ran everything. This is the kind of unfairness that is easy to miss and "
 "changes results.", AMBER, colors.HexColor("#E0BE84")))

s.append(P("7. The five ways of combining (the \"ladder\")", H1))
s.append(P("We did not build just one combiner. We built five, from simplest to most complex, "
           "so we could see <b>which part actually helps</b>. If only the complicated one had been "
           "built, we would never have known how little the complicated one adds.", BODY))
s.append(table([
 ["Name", "What it does", "Size (settings it learns)"],
 ["concat", "Just glues the three descriptions end to end. No thinking.", "0"],
 ["gated", "Learns how much to trust each view, per molecule. Like a volume knob per view.", "16,513"],
 ["xattn", "Lets the views \"talk to each other\" and compare notes.", "1,054,208"],
 ["bilinear", "Multiplies views together to catch combined effects.", "99,072"],
 ["proposed", "All three of the above, stacked. Our most complex design.", "1,169,793"],
], [24*mm, 106*mm, 38*mm]))
s.append(P("The last column is roughly \"how many dials the program can adjust\". The "
           "<b>proposed</b> combiner has about <b>71 times more dials</b> than the simple "
           "<b>gated</b> one (fusion step only). Remember that number for the results section.", SMALL))

s.append(PageBreak())

# ───────────────────────── PAGE 4 ─────────────────────────
s.append(P("8. How the data is split (very important)", H1))
s.append(P("You cannot test a student using the exact questions they revised from. Same for a "
           "computer program. So we cut each dataset into three parts:", BODY))
s.append(table([
 ["Part", "Share", "What it is used for", "Can the model see the answers?"],
 ["Train", "80%", "Learning the patterns.", "Yes, this is the textbook."],
 ["Validation", "10%", "Deciding when to stop learning, and picking settings.", "Yes, this is the mock exam."],
 ["Test", "10%", "The final score we report.", "<b>No. Touched once, at the very end.</b>"],
], [24*mm, 16*mm, 76*mm, 52*mm]))

s.append(P("Why we split by \"scaffold\" and not randomly", H2))
s.append(P("A <b>scaffold</b> is the core skeleton of a molecule. Many molecules share one skeleton "
           "with small bits changed.", BODY))
s.append(B("If we split <b>randomly</b>, near-identical molecules land in both train and test. The "
           "program is basically being tested on what it memorised, and scores look great but are fake."))
s.append(B("If we split <b>by scaffold</b>, all molecules sharing a skeleton go to the same side. "
           "Now the test contains genuinely <i>new</i> shapes. Scores drop, but they are honest."))
s.append(B("This matches real life: a chemist wants to predict something <b>new</b>, not something "
           "already in the books."))

s.append(P("Why we do this six times, not once", H2))
s.append(box(
 "One split is one roll of the dice. Do it a different way and the score changes. Two procedures that "
 "are both called \"scaffold split\" gave the same model on the same data scores up to <b>0.22 AUC "
 "apart</b>, which is larger than almost any improvement papers claim. So we use <b>6 different splits</b> "
 "(1 standard + 5 shuffled) and report the <b>average with a range</b>. A result that only shows up "
 "in one split is not a result.", AMBER, colors.HexColor("#E0BE84")))

s.append(PageBreak())
s.append(P("9. How training actually works", H1))
s.append(P("Training is repetition with correction. Here is one full cycle, called an <b>epoch</b>:", BODY))
s.append(table([
 ["Step", "What happens"],
 ["1", "Take a small batch of 128 molecules from the <b>train</b> part."],
 ["2", "The program guesses the answer for each one."],
 ["3", "Compare guesses with the real answers and measure how wrong it was. This number is the <b>loss</b>."],
 ["4", "Nudge all the internal dials slightly in the direction that would have been less wrong."],
 ["5", "Repeat for the next batch, until all training molecules have been seen once. That is one epoch."],
 ["6", "Now test on the <b>validation</b> part. Did it improve?"],
 ["7", "If it keeps improving, run another epoch. If it stops improving for 15 epochs in a row, <b>stop</b>."],
 ["8", "Go back to the best epoch, and only now run the <b>test</b> part. That score is reported."],
], [14*mm, 154*mm]))
s.append(P("Step 7 is called <b>early stopping</b>. Without it the program starts memorising the "
           "training molecules instead of learning general patterns, which is called "
           "<b>overfitting</b> &mdash; like a student who memorises answers but cannot handle a new "
           "question.", BODY))
s.append(P("One more rule we follow: some molecules have <b>missing</b> answers, because nobody ran "
           "that lab test. We skip those cells. The original code treated missing as \"not toxic\", "
           "which invented fake answers for <b>24% of the Tox21 test labels</b>. Finding and fixing that "
           "changed the results.", BODY))

s.append(PageBreak())

# ───────────────────────── PAGE 5 ─────────────────────────
s.append(P("10. Technology stack", H1))
s.append(P("Everything is written in <b>Python</b>. Each tool below does one job.", BODY))
s.append(table([
 ["Tool", "Version", "What it does for us, in one line"],
 ["Python", "3.11", "The programming language everything is written in."],
 ["PyTorch", "2.3.1", "The main engine for building and training the models."],
 ["PyTorch Geometric", "2.6.1", "An add-on that lets PyTorch handle molecules-as-networks."],
 ["RDKit", "2025.3.5", "The chemistry toolkit. Reads molecule strings, calculates chemical facts."],
 ["DeepChem", "2.8.0", "Supplies the 8 standard datasets and the standard scaffold split."],
 ["Transformers", "4.43.4", "Runs ChemBERTa, the molecule-reading language model."],
 ["PEFT (LoRA)", "0.12.0", "Lets us fine-tune the big language model cheaply, by training only a "
  "tiny part of it."],
 ["scikit-learn", "1.4.2", "Classic machine learning: Random Forest, and the scoring functions."],
 ["SciPy / NumPy / pandas", "&mdash;", "Maths, statistics and table handling."],
 ["Chemprop", "2.3.1", "An outside published model, used as an independent yardstick."],
 ["Optuna", "4.5.0", "A settings-tuner. Written into the project but deliberately never run &mdash; see Q8."],
 ["Git &amp; GitHub", "&mdash;", "Keeps every version of the code and results, so nothing is lost."],
 ["Google Colab", "&mdash;", "Free cloud computers with GPUs, for the long training runs."],
], [40*mm, 22*mm, 106*mm]))
s.append(P("A <b>GPU</b> is a chip originally made for video games. It does thousands of small sums "
           "at once, which is exactly what training needs. Our longest runs used one in Google Colab.", SMALL))

s.append(P("11. What we found", H1))
s.append(P("These are the real results, stated plainly.", BODY))
s.append(table([
 ["Question we asked", "Honest answer"],
 ["Did a hidden clue in the data inflate the text-reading models?",
  "<b>Yes.</b> In two datasets the way a molecule is <i>written</i> gives the answer away. Removing "
  "the clue dropped the text model from 0.99 to 0.80 on one of them."],
 ["Does our combined model beat the simple network we started from?",
  "<b>Not convincingly.</b> It had the better average on 7 of 8 datasets, but that is not enough "
  "to rule out luck, and only 1 dataset passed the strict check."],
 ["Does it beat a simple list of chemical facts (the descriptor view)?",
  "<b>No.</b> It won on 3 of 8, and every other combiner did worse than it on 6 or 7 of 8."],
 ["Is the complex combiner better than the cheap one?",
  "<b>Not clearly.</b> It averaged better on 6 of 8 datasets, but no single dataset passed the "
  "strict check."],
 ["Do we need the molecule-as-network view at all?",
  "<b>Apparently not.</b> Removing it made no detectable difference and ran <b>13 times faster</b>, "
  "though we could only prove the two are equivalent on 2 of 8 datasets."],
 ["Does a famous published model (AttentiveFP) beat the old simple one?",
  "<b>No</b>, not reliably &mdash; at about 12 times the size. So this is not just our model being weak."],
 ["Does training on a different computer give the same answer?",
  "<b>Not exactly.</b> Moving from a laptop CPU to a GPU moved about a quarter of single-split "
  "scores by more than our threshold &mdash; about as much as changing the random seed."],
], [72*mm, 96*mm]))
s.append(box("<b>The honest headline:</b> once a hidden clue in the data was removed, our model "
             "was no longer clearly better than the simple network it started from, and it never "
             "beat a simple list of chemical facts. We report both.", GREEN, colors.HexColor("#9CC3A9")))

s.append(PageBreak())

# ───────────────────────── PAGE 6 ─────────────────────────
s.append(P("12. The \"how sure are you?\" part", H1))
s.append(P("A prediction of \"toxic\" is not much use without knowing how confident the program is. "
           "So we added a method called <b>conformal prediction</b>.", BODY))
s.append(B("Instead of one answer, it gives a <b>set</b> of answers with a promise: \"the truth is in "
           "here 90% of the time\"."))
s.append(B("If the program is unsure it returns both \"toxic\" and \"safe\", which is its way of "
           "saying <i>I do not know</i>. That is more useful than a confident wrong answer."))
s.append(P("We found a serious problem with the standard version:", BODY))
s.append(box(
 "It keeps its 90% promise <b>overall</b> &mdash; but only because most molecules are safe. For the "
 "<b>toxic</b> ones, the ones you actually care about, it was right as little as <b>10% of the time</b> "
 "in the worst case. We showed the cause: models trained without extra weight on the rare toxic class "
 "fail this way, and the same model trained with that weight does not. A fix called "
 "<b>class-conditional</b> conformal prediction repairs it, by making a separate promise for each answer.",
 AMBER, colors.HexColor("#E0BE84")))
s.append(P("<b>Be ready for this one:</b> this failure was already known, and another team measured it "
           "on these benchmarks in July 2026. We cite them and present our version "
           "as <b>confirmation on more models, plus the class-weighting experiment</b>, rather than as our discovery. Saying "
           "this openly is the correct thing to do.", BODY))

s.append(P("13. Questions a teacher may ask", H1))

qa = [
 ("Q1. What is actually new here, if the model did not win?",
  "The strict testing method, and what it revealed. We found 7 real errors in the system we "
  "inherited, a hidden clue in two standard datasets, and no evidence that the popular fusion ideas we "
  "tested beat a simple baseline. A carefully proven "
  "\"this does not work\" is useful, because it saves other people from repeating it."),
 ("Q2. Why is a negative result worth anything?",
  "Because we can show it is trustworthy. We tested on 8 datasets, 6 splits each, with statistical "
  "correction. Most papers use 1 or 2 datasets and 1 split, where luck can look like success."),
 ("Q3. How do you know your code is not simply broken?",
  "We tested the parts separately. We checked the combining step really does mix the views, by "
  "proving the simplest one is exactly additive and the complex one is not. We also checked our "
  "uncertainty method on made-up data with known answers first, and it was correct there."),
 ("Q4. Why 8 datasets, why not 3?",
  "Statistical strength comes from the number of datasets. With 8, the best possible confidence "
  "score is 0.0078, which is strong. With 5 it could never reach the usual 0.05 threshold at all."),
 ("Q5. What is overfitting, and how did you prevent it?",
  "It is memorising instead of learning. We prevented it with early stopping, by splitting by "
  "scaffold so the test is genuinely new, and by never letting the model see the test data until "
  "the very end."),
 ("Q6. What is data leakage? Did you have any?",
  "It is when the answer sneaks into the input, making scores fake. We found two kinds. The inherited "
  "system picked its best model by looking at the test scores; we fixed that so all choices are made "
  "on the validation part only. And in ClinTox and BBBP the way a molecule is written gives away the "
  "answer to a text-reading model; we now rewrite the molecules of those two datasets in one standard form first."),
 ("Q7. Why did you use a language model on chemistry?",
  "Chemists already write molecules as text strings. A language model trained on millions of those "
  "strings learns chemical patterns the same way a text model learns grammar."),
 ("Q8. Did you tune the settings to get the best scores?",
  "Deliberately not. Every model uses one identical setting. If we tuned only our model and compared "
  "it to untuned ones, we would be measuring effort, not quality. We wrote the tuning tool and left "
  "it switched off on purpose, and we state this as a limitation."),
 ("Q9. What is the single most useful thing you found?",
  "That the way a molecule is written can leak the answer. Text-reading models looked much better on "
  "two standard datasets than they really are. Anyone using a language model on these benchmarks "
  "should rewrite the molecules in one standard form first."),
 ("Q10. What would you do with more time?",
  "Run the settings-tuner fairly on every model, add more outside comparison models, and test on "
  "harder real-world data rather than public benchmark sets."),
 ("Q11. Can this be used in a real company?",
  "The predictor is not better than simple chemistry, so not as a product yet. The testing method "
  "is usable immediately \u2014 any team can run it to check whether their own model is really working."),
 ("Q12. How big is the project?",
  "8 datasets, 6 splits, more than 20 model variants compared, training on a laptop CPU and cloud "
  "GPUs, and an automatic checker that re-verifies every table and the key numbers in the write-up "
  "against the saved results."),
]
for q, a in qa:
    s.append(KeepTogether([Paragraph(q, QSTYLE), Paragraph(a, ASTYLE)]))

s.append(Spacer(1, 10))
s.append(box("<b>If you remember one sentence:</b> we built a fair way to test molecule-prediction "
             "programs, used it on our own, and it told us our complicated idea was no "
             "better than a simple one &mdash; which is exactly what a good measuring tool is "
             "supposed to do.", LIGHT, BORDER))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=21*mm, rightMargin=21*mm,
                        topMargin=18*mm, bottomMargin=20*mm,
                        title="Molecular Property Prediction - Project Explained",
                        author="")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, os.path.getsize(OUT)//1024, "KB")
