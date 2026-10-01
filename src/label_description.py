"""src/label_descriptions.py

Maps raw candidate labels -> descriptive candidate strings.
Order of candidates is preserved, so the integer labels stay valid.
"""


def _norm(label: str) -> str:
    return label.strip().lower().replace("_", " ")


EMOTION = {
    "sadness": "sadness: the person feels unhappy, sorrowful, depressed, disappointed, or emotionally down",
    "joy": "joy: the person feels happy, cheerful, delighted, content, or full of enjoyment",
    "love": "love: the person feels affection, care, warmth, tenderness, or romantic attachment toward someone",
    "anger": "anger: the person feels angry, irritated, annoyed, resentful, or furious",
    "fear": "fear: the person feels afraid, anxious, worried, nervous, or threatened",
    "surprise": "surprise: the person feels surprised, shocked, amazed, or caught off guard by something unexpected",
}

_WORLD = "world news: international politics, countries, governments, conflicts, diplomacy, and global events"
_SPORTS = "sports news: games, matches, athletes, teams, tournaments, and competitions"
_BUSINESS = "business news: companies, markets, the economy, finance, trade, and corporate earnings"
_SCITECH = ("science and technology news: scientific research, computing, the internet, software, "
            "gadgets, space, and technology companies")

AG_NEWS = {
    "world": _WORLD,
    "sports": _SPORTS,
    "business": _BUSINESS,
    "sci/tech": _SCITECH,
    "science/technology": _SCITECH,
    "science and technology": _SCITECH,
    "sci tech": _SCITECH,
}

# Banking77: description only, the "label: " prefix is added below.
_BANKING = {
    "activate my card": "the customer wants to activate a new card they received",
    "age limit": "the customer asks about the minimum age required to open or use an account",
    "apple pay or google pay": "the customer asks about using Apple Pay or Google Pay with their card",
    "atm support": "the customer asks which ATMs they can use or needs help with an ATM",
    "automatic top up": "the customer asks about automatic top-ups of their account balance",
    "balance not updated after bank transfer": "the customer made a bank transfer but the balance has not been updated",
    "balance not updated after cheque or cash deposit": "the customer deposited a cheque or cash but the balance has not been updated",
    "beneficiary not allowed": "the customer cannot add or send money to a beneficiary because it is not allowed",
    "cancel transfer": "the customer wants to cancel a transfer they already made",
    "card about to expire": "the customer's card is about to expire and they ask about renewal",
    "card acceptance": "the customer asks where their card is accepted",
    "card arrival": "the customer asks where their ordered card is or when it will arrive",
    "card delivery estimate": "the customer asks how long card delivery will take",
    "card linking": "the customer wants to link a card to their account or app",
    "card not working": "the customer's card is not working or is not being accepted",
    "card payment fee charged": "the customer was charged a fee for a card payment",
    "card payment not recognised": "the customer sees a card payment on their account that they do not recognise",
    "card payment wrong exchange rate": "the customer was given the wrong exchange rate on a card payment",
    "card swallowed": "an ATM kept or swallowed the customer's card",
    "cash withdrawal charge": "the customer was charged a fee while withdrawing cash from an ATM",
    "cash withdrawal not recognised": "the customer sees a cash withdrawal on their account that they did not make",
    "change pin": "the customer wants to change or reset their card PIN",
    "compromised card": "the customer believes their card details were compromised or used fraudulently",
    "contactless not working": "the customer's contactless payments are not working",
    "country support": "the customer asks which countries the service or card is available in",
    "declined card payment": "a card payment made by the customer was declined",
    "declined cash withdrawal": "an ATM cash withdrawal by the customer was declined",
    "declined transfer": "a money transfer made by the customer was declined",
    "direct debit payment not recognised": "the customer sees a direct debit payment they do not recognise",
    "disposable card limits": "the customer asks about spending limits on disposable virtual cards",
    "edit personal details": "the customer wants to change personal details such as name, address, or phone number",
    "exchange charge": "the customer asks about fees charged for currency exchange",
    "exchange rate": "the customer asks what exchange rate is used for currency conversion",
    "exchange via app": "the customer asks how to exchange currencies using the app",
    "extra charge on statement": "the customer sees an unexpected extra charge on their statement",
    "failed transfer": "a money transfer by the customer failed",
    "fiat currency support": "the customer asks which fiat currencies are supported",
    "get disposable virtual card": "the customer wants to get a disposable virtual card",
    "get physical card": "the customer wants to get a physical card",
    "getting spare card": "the customer wants an additional or spare card",
    "getting virtual card": "the customer wants to get a virtual card",
    "lost or stolen card": "the customer's card was lost or stolen",
    "lost or stolen phone": "the customer's phone was lost or stolen and they worry about account security",
    "order physical card": "the customer wants to order a physical card",
    "passcode forgotten": "the customer forgot their passcode and needs to recover it",
    "pending card payment": "a card payment is still pending and has not completed",
    "pending cash withdrawal": "a cash withdrawal is still pending and has not completed",
    "pending top up": "a top-up is still pending and has not been credited",
    "pending transfer": "a transfer is still pending and has not completed",
    "pin blocked": "the customer's PIN is blocked after too many wrong attempts",
    "receiving money": "the customer asks how to receive money into their account",
    "refund not showing up": "the customer expected a refund but it has not appeared in their account",
    "request refund": "the customer wants to request a refund for a payment",
    "reverted card payment?": "a card payment was reverted or cancelled and the customer asks about it",
    "supported cards and currencies": "the customer asks which cards and currencies are supported",
    "terminate account": "the customer wants to close or terminate their account",
    "top up by bank transfer charge": "the customer was charged a fee for topping up by bank transfer",
    "top up by card charge": "the customer was charged a fee for topping up by card",
    "top up by cash or cheque": "the customer asks about topping up their account with cash or a cheque",
    "top up failed": "the customer's top-up attempt failed",
    "top up limits": "the customer asks about limits on how much they can top up",
    "top up reverted": "a top-up was reverted or reversed after being made",
    "topping up by card": "the customer asks how to top up their account using a card",
    "transaction charged twice": "the customer was charged twice for the same transaction",
    "transfer fee charged": "the customer was charged a fee for a money transfer",
    "transfer into account": "the customer asks how to transfer money into their account",
    "transfer not received by recipient": "the recipient of the customer's transfer has not received the money",
    "transfer timing": "the customer asks how long a transfer takes",
    "unable to verify identity": "the customer is unable to complete identity verification",
    "verify my identity": "the customer asks how to verify their identity",
    "verify source of funds": "the customer is asked to verify the source of their funds",
    "verify top up": "the customer needs to verify a top-up",
    "virtual card not working": "the customer's virtual card is not working",
    "visa or mastercard": "the customer asks whether their card is Visa or Mastercard",
    "why verify identity": "the customer asks why they need to verify their identity",
    "wrong amount of cash received": "the customer received the wrong amount of cash from an ATM",
    "wrong exchange rate for cash withdrawal": "the customer was given the wrong exchange rate on a cash withdrawal",
}
BANKING77 = {k: f"{k}: {v}" for k, v in _BANKING.items()}

# Labels are distinct across datasets, so one flat lookup is enough.
DESCRIPTIONS = {**EMOTION, **AG_NEWS, **BANKING77}


GROUPS = [EMOTION, AG_NEWS, BANKING77]


def describe_candidates(candidates):
    norm = [_norm(c) for c in candidates]
    missing = [c for c, n in zip(candidates, norm) if n not in DESCRIPTIONS]
    group = next((g for g in GROUPS if norm and norm[0] in g), None)
    used = {DESCRIPTIONS[n] for n in norm if n in DESCRIPTIONS}
    unused = sorted(set(group.values()) - used) if group else []
    if missing or unused:
        raise ValueError(
            f"Class/description mismatch.\n"
            f"  dataset classes with no description: {missing}\n"
            f"  descriptions with no matching dataset class: {unused}"
        )
    return [DESCRIPTIONS[n] for n in norm]