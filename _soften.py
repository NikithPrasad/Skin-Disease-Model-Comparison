"""One-off: gentler wording for doctor advice on both sites (same meaning, kinder tone)."""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DET, CMP = BASE / "Skin-Disease-Detection", BASE / "Skin-Disease-Model-Comparison"

predictor = [
    ('#   level      "urgent" (see a doctor soon), "doctor" (book an appointment), "selfcare" (usually harmless)',
     '#   level      "urgent" (please talk to a doctor), "doctor" (worth talking to a doctor), "selfcare" (usually harmless)'),
    ('"headline": "Please see a doctor soon, ideally within 2 weeks",',
     '"headline": "This could be a melanoma, so we\'d kindly ask you to talk to a doctor",'),
    ('"Book an appointment with a GP or dermatologist and show them this spot.",',
     '"When you can, please book an appointment with a GP or dermatologist and show them this spot. It\'s best not to wait more than a couple of weeks.",'),
    ('"Do not try to remove, cut, burn or treat it at home.",',
     '"Please don\'t try to remove or treat it at home.",'),
]
headline_doctor = ('"headline": "Book a doctor\'s appointment in the next few weeks",',
                   '"headline": "This may need treatment, so we\'d suggest talking to a doctor when you can",')

js_common = [
    ('["warn", "Can be serious"]', '["warn", "Worth checking"]'),
    ('["warn", "Get it checked"]', '["warn", "Worth checking"]'),
    ('el("h3", { text: "See a doctor straight away if" })', 'el("h3", { text: "Please talk to a doctor if you notice that" })'),
]
js_det = [
    ('level = "urgent"; headline = "Please see a doctor soon: melanoma can\'t be ruled out";',
     'level = "urgent"; headline = "A melanoma can\'t be fully ruled out, so we\'d kindly ask you to talk to a doctor";'),
    ('''    why = `Although ${info.name.toLowerCase()} is the most likely answer, the model gives melanoma a ${pct(pMel)} chance. ` +
      `This site asks you to see a doctor whenever that chance is ${pct(m.melanoma_threshold)} or more, which in testing caught ${whole(m.test.melanoma_recall_with_warning)} of melanomas.`;''',
     '''    why = `This is most likely ${info.name.toLowerCase()}, but there is a ${pct(pMel)} chance it could be a melanoma. Just to be safe, ` +
      `we'd recommend having a doctor look at it. (The site suggests this whenever that chance is ${pct(m.melanoma_threshold)} or more, which in testing caught ${whole(m.test.melanoma_recall_with_warning)} of melanomas.)`;'''),
    ('level = "doctor"; headline = "Have a doctor take a look";\n    why = "The model is not confident about this photo, so a doctor\'s opinion is the safe next step.";',
     'level = "doctor"; headline = "We\'re not sure about this one, so we\'d suggest asking a doctor";\n    why = "The model isn\'t confident about this photo. A doctor can tell you for sure.";'),
]
js_cmp = [
    ('level = "urgent"; headline = "Please see a doctor soon: melanoma can\'t be ruled out";',
     'level = "urgent"; headline = "A melanoma can\'t be fully ruled out, so we\'d kindly ask you to talk to a doctor";'),
    ('why = `Most models say ${info.name.toLowerCase()}, but ${melVotes} of ${total} ${melVotes === 1 ? "thinks" : "think"} this could be a melanoma, so it is worth having checked.`;',
     'why = `Most models say ${info.name.toLowerCase()}, but ${melVotes} of ${total} ${melVotes === 1 ? "thinks" : "think"} it could be a melanoma. Just to be safe, we\'d recommend having a doctor look at it.`;'),
    ('level = "doctor"; headline = "Have a doctor take a look";\n    why = "The models disagree about this photo, so a doctor\'s opinion is the safe next step.";',
     'level = "doctor"; headline = "We\'re not sure about this one, so we\'d suggest asking a doctor";\n    why = "The models don\'t agree about this photo. A doctor can tell you for sure.";'),
]
html_common = [
    ("Every check ends with what to do next, from self-care to seeing a doctor.",
     "Every check ends with what to do next, from self-care to talking to a doctor."),
    ("and whether to see a doctor.</span></li>", "and whether it's worth talking to a doctor.</span></li>"),
    ("Anyone worried about a skin lesion should see a doctor or dermatologist.",
     "If a skin spot worries you, we'd recommend talking to a doctor or dermatologist."),
]


def apply(path, reps, expect_all=True):
    t = path.read_text(encoding="utf-8")
    for a, b in reps:
        n = t.count(a)
        assert n >= 1 or not expect_all, (path.name, a[:60])
        t = t.replace(a, b)
    path.write_text(t, encoding="utf-8")


for site in [DET, CMP]:
    t = (site / "app/predictor.py").read_text(encoding="utf-8")
    assert t.count(headline_doctor[0]) == 2, site
    apply(site / "app/predictor.py", predictor + [headline_doctor])
    apply(site / "app/static/app.js", js_common + (js_det if site == DET else js_cmp))
    apply(site / "app/static/index.html", html_common)
print("softened both sites")
