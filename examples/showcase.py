"""The library talking in plain English, no LLM needed. Run: python examples/showcase.py"""

from bike_setup import ex, test

week_before, thanksgiving = test.iloc[[7617]], test.iloc[[7785]]   # Thursday 8am, Nov 2012
dry_wednesday, rainy_wednesday = test.iloc[[7111]], test.iloc[[7243]]  # Wednesday 5pm, Sep/Oct 2012


def section(question, result):
    print(f"\n▶ {question}\n")
    print(result)


section("Why does Thanksgiving morning's forecast collapse compared with the Thursday before?",
        ex.explain_change(week_before, thanksgiving))
section("Is 63 rentals unusual?", ex.prediction_context(thanksgiving))
section("What if Thanksgiving were a normal working day?",
        ex.what_if(thanksgiving, {"holiday": "False", "workingday": "True"}))
section("Why is this Wednesday evening forecast so much lower than last week's?",
        ex.explain_change(dry_wednesday, rainy_wednesday))
section("Why is the forecast for that rainy evening what it is?", ex.explain_prediction(rainy_wednesday))
section("What matters most on weekends and holidays?", ex.top_drivers({"workingday": "False"}))
section("How good is the model, and where is it worst?", ex.model_errors(by="season"))
section("Will lowering prices increase rentals?",
        ex.flag_unanswerable("causal", "asks what a real-world action would cause"))
