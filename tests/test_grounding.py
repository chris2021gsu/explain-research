from explainkit import unsupported_numbers


def test_rounding_and_suffixes_are_allowed():
    src = ['{"change": 4987, "pct_change": 12.47, "effect": -3204}']
    assert unsupported_numbers("Up about 5k (12.5%), mostly demand (-3,200).", src) == []
    assert unsupported_numbers("It rose by 4,987 rentals.", src) == []


def test_invented_numbers_are_flagged():
    src = ['{"change": 4987}']
    assert unsupported_numbers("It rose by 7,300.", src) == ["7,300"]


def test_numbers_from_the_question_are_allowed():
    assert unsupported_numbers("From 35k to 40k.", ["Why did it go from 35000 to 40000?"]) == []


def test_small_counts_are_ignored():
    assert unsupported_numbers("The top 3 factors explain it.", []) == []
