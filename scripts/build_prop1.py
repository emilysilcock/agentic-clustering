"""Build proposer-1 proposal for banking77 seed=0_auditdiv.

Angle: ~65 clusters, lean toward grouping when actions are very close.
Style: 'broad-action'. Names align with the user's "specific action or information
the customer is asking about" framing.
"""
import json
import os
import sys
import uuid
import datetime as dt
from pathlib import Path

WS = Path(os.environ["CLUSTERING_WORKSPACE"])
SAMPLE = WS / "samples" / "shared_proposer_sample.json"
OUT_DIR = WS / "proposals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with SAMPLE.open("r", encoding="utf-8") as f:
    sample = json.load(f)

ids_in_sample = [r["id"] for r in sample]
id_set = set(ids_in_sample)
assert len(ids_in_sample) == 600, f"Expected 600 texts, got {len(ids_in_sample)}"

# Short id -> full id map (the last 6 digits are unique within banking77-test)
short = {r["id"].rsplit("-", 1)[-1]: r["id"] for r in sample}

def ids(*shorts):
    out = []
    for s in shorts:
        if s not in short:
            raise KeyError(f"short id {s!r} not found in sample")
        out.append(short[s])
    return out

CLUSTERS = [
    # ---- CARD DELIVERY / ARRIVAL ----
    {
        "name": "Card delivery status and arrival delays",
        "description": "Customer asks where their newly ordered physical card is, how long delivery takes, or expresses concern that it has not arrived yet. Also covers tracking numbers and arrival timing questions.",
        "shorts": ["000004", "000302", "000282", "000312", "000013", "000009", "000036", "000032", "000006", "000039", "000286", "000290", "000303", "000294", "000000"],
        "reasoning": "Per angle: merge 'card not arrived yet' + 'how long for card to arrive' + delivery-time + tracking-number + 'how do I locate my card?' (treated as where-is-it) into one delivery-status intent.",
    },
    {
        "name": "Where can I receive my physical card",
        "description": "Customer asks which addresses or countries a physical card can be shipped to. Differs from delivery-status because it is forward-looking about destination eligibility, not about an in-flight order.",
        "shorts": ["002506", "002509", "002480", "002483", "002491"],
        "reasoning": "Delivery destination questions are conceptually distinct from arrival-timing questions.",
    },
    {
        "name": "Request or order a physical card",
        "description": "Customer asks how to obtain, request, or order a physical (non-virtual) card, including replacements when current card is about to expire.",
        "shorts": ["002502", "002488", "002497", "002499", "002415", "002947", "002943", "002926", "002925", "002923", "002959", "002927", "002931", "002946", "002505", "002507"],
        "reasoning": "Combines plain physical-card requests with renewal-before-expiry requests since both ask the same workflow. Includes physical-card fee questions because they are inseparable from the request itself in surface form.",
    },
    {
        "name": "Country and region eligibility for the service",
        "description": "Customer asks whether the service or card is available in their country / region, or where the company operates. Geographic eligibility for opening or holding an account.",
        "shorts": ["003059", "003045", "003048", "003054", "003056", "003058", "003069", "003061"],
        "reasoning": "All are 'do you support my country?' questions.",
    },

    # ---- VIRTUAL / DISPOSABLE CARDS ----
    {
        "name": "Get or order a virtual card",
        "description": "Customer asks how to obtain a virtual card, including reissuing one. Excludes disposable-virtual-card explanation.",
        "shorts": ["000921", "000956"],
        "reasoning": "Plain virtual-card request — kept separate from disposable since the action verb differs.",
    },
    {
        "name": "How disposable virtual cards work",
        "description": "Customer asks for an explanation or description of disposable virtual cards: what they are, how they work, how to get one.",
        "shorts": ["002623", "002626", "002600", "002602", "002610", "002616", "002622", "002638"],
        "reasoning": "Conceptual/how-to questions about disposable virtual cards.",
    },
    {
        "name": "Disposable card usage limits",
        "description": "Customer asks about limits on disposable cards: how many they can have, max per day, transactions per card, multi-use restrictions.",
        "shorts": ["001364", "001362", "001373", "001369", "001365", "001382", "001385", "001380", "001368", "001393", "001372", "001375", "001383"],
        "reasoning": "All about quantity/usage limits for disposable cards.",
    },

    # ---- EXTRA / FAMILY CARDS, LINKING ----
    {
        "name": "Request additional or extra cards",
        "description": "Customer asks for additional cards on the same account, including for family members. Includes asking about fees for extra cards.",
        "shorts": ["002406", "002417", "002404", "002435", "002401"],
        "reasoning": "Extra-card requests, including the dependent fee question.",
    },
    {
        "name": "Link an external card to the app",
        "description": "Customer wants to link or connect a new external credit/debit card to the app or service.",
        "shorts": ["000054", "000050", "000060", "000057", "000058", "000066"],
        "reasoning": "Linking/viewing-a-linked-card actions — all map to the same setup workflow.",
    },

    # ---- SUPPORTED CARDS / CARD CHOICE ----
    {
        "name": "Which card brands and currencies are supported",
        "description": "Customer asks which card networks (Visa, Mastercard, Amex) and currencies are supported by the service.",
        "shorts": ["001295", "001310", "001298", "001289", "001303", "001318", "000896", "000904", "000905", "000889"],
        "reasoning": "Visa/Mastercard preference and supported-card questions go together with currency-support questions because they recur in the same utterance pattern.",
    },

    # ---- ACTIVATE CARD ----
    {
        "name": "Activate a card",
        "description": "Customer asks how to activate their card, or requests activation. Includes how to start using a new card.",
        "shorts": ["002864", "002857", "002868", "002871", "002879", "002844", "002872"],
        "reasoning": "All activation requests.",
    },

    # ---- CARD NOT WORKING / DECLINED ----
    {
        "name": "Card not working — general diagnosis",
        "description": "Customer reports their card is not working, broken, stopped working, or asks why a card is failing in a non-specific way. Includes generic 'help with my card' diagnostic requests.",
        "shorts": ["000388", "000382", "000374", "000383", "000375", "000367", "000371", "000373", "002544", "001835"],
        "reasoning": "Generic 'card broken / not working / unspecified failure' bucket — kept broad per angle.",
    },
    {
        "name": "Card payment declined or not completing",
        "description": "Customer reports a card payment was declined, stopped, cancelled, or otherwise did not complete. Includes new-card decline-on-first-use complaints.",
        "shorts": ["001813", "001822", "001834", "002080", "002083", "002115", "001824", "001811", "001828", "001837", "002094", "000599"],
        "reasoning": "Card-not-present / point-of-sale payment failure — distinct from ATM-decline and from generic card-broken. Includes 'app refused approved payment' and 'should I reinstall the payment app' as adjacent payment-failure complaints.",
    },
    {
        "name": "ATM declined card or won't dispense cash",
        "description": "Customer reports an ATM declined them, didn't give cash, or the card was declined at the ATM specifically.",
        "shorts": ["001577", "001576", "001583", "001589", "001591", "001573", "001565", "001571", "001570"],
        "reasoning": "ATM-specific decline issues — physical-machine context.",
    },
    {
        "name": "Contactless or NFC not working",
        "description": "Customer reports contactless / NFC / tap-to-pay is not working.",
        "shorts": ["000570", "000572", "000567", "000583", "000577"],
        "reasoning": "Tight contactless-payment failure cluster.",
    },
    {
        "name": "Virtual card not working / declined",
        "description": "Customer reports their virtual or disposable virtual card was declined, rejected, or won't work.",
        "shorts": ["002532", "002541", "002543", "002553"],
        "reasoning": "Virtual-card-specific failure intents.",
    },

    # ---- LOST / STOLEN / COMPROMISED ----
    {
        "name": "Lost or stolen card and replacement",
        "description": "Customer reports their card is lost or stolen and wants help blocking it, reporting it, and / or getting a replacement.",
        "shorts": ["000446", "000479", "000475", "000470", "000471", "000445", "000472", "000454", "000474"],
        "reasoning": "All lost/stolen-card incidents — replacement requests bundled per angle.",
    },
    {
        "name": "Lost or stolen phone — secure account",
        "description": "Customer lost their phone or had it stolen and wants to secure the account or prevent access. Distinct from lost-card because the device, not the card, is the vector.",
        "shorts": ["001658", "001647", "001645", "001651", "001662", "001671", "001678"],
        "reasoning": "Device-loss security intent.",
    },
    {
        "name": "Unrecognized payment or fraud suspected",
        "description": "Customer sees an unfamiliar charge, payment, or direct debit on their statement and suspects fraud, an unauthorized purchase, or a hacked account.",
        "shorts": ["001110", "001111", "001402", "001419", "001434", "001516", "001502", "001508", "001481", "001504"],
        "reasoning": "Unauthorized-charge / dispute-a-charge family.",
    },
    {
        "name": "Unrecognized ATM cash withdrawal",
        "description": "Customer sees an ATM cash withdrawal in the app or statement that they did not make.",
        "shorts": ["002736", "002726", "002737", "002744", "002725", "002755", "002758", "002759", "002724", "002756", "002732", "002739"],
        "reasoning": "Distinct from card-fraud because the surface form is always 'a cash withdrawal I didn't do' — keep as fine intent.",
    },
    {
        "name": "Found lost card — reactivate or unblock",
        "description": "Customer found a previously lost card and wants to reactivate or unblock it.",
        "shorts": ["000076", "000062"],
        "reasoning": "Reactivation after recovery.",
    },
    {
        "name": "Freeze or block card",
        "description": "Customer explicitly asks to freeze, block, or lock their card via the app — typically a precaution rather than a confirmed loss.",
        "shorts": ["001421", "001407"],
        "reasoning": "Card-freeze action.",
    },

    # ---- ATM PHYSICAL ISSUES ----
    {
        "name": "Card swallowed or trapped in ATM",
        "description": "Customer reports an ATM has swallowed, trapped, or refused to return their physical card and asks how to recover it.",
        "shorts": ["001952", "001933", "001953", "001951", "001931", "001926", "001948", "001928", "001941", "001958"],
        "reasoning": "Stuck-in-ATM physical recovery scenario.",
    },
    {
        "name": "Wrong cash dispensed at ATM",
        "description": "Customer reports the ATM gave a different amount of cash than requested (typically less), or the wrong currency amount appears.",
        "shorts": ["000782", "000775", "000764", "000779", "000797", "000787", "000770", "000780", "000786", "000772", "000798"],
        "reasoning": "Cash-dispensed-mismatch — the customer got the wrong amount in hand.",
    },
    {
        "name": "Find or eligibility of nearby ATMs",
        "description": "Customer asks where they can use their card at an ATM, where the nearest ATM is, or which ATMs accept the card.",
        "shorts": ["001466", "001449", "001463", "001442", "001462", "001468", "001471", "001478", "001444"],
        "reasoning": "ATM-locator and ATM-compatibility queries.",
    },

    # ---- ACCEPTED MERCHANTS ----
    {
        "name": "Where can my card be used / accepted merchants",
        "description": "Customer asks which merchants, stores, or contexts accept the card, including online use and country-limited acceptance.",
        "shorts": ["000999", "000977", "000964", "000995", "000980", "000983", "000981", "000978", "000991"],
        "reasoning": "Merchant acceptance / payment-channel acceptance — kept broad.",
    },

    # ---- PIN ----
    {
        "name": "Find or view PIN",
        "description": "Customer asks where to find, view, or look up their current card PIN.",
        "shorts": ["001270", "001278", "001261", "000521", "001240", "001242", "001275"],
        "reasoning": "Find-my-PIN intent.",
    },
    {
        "name": "Change or reset PIN",
        "description": "Customer wants to change, update, set a new, or reset their card PIN, including doing so while abroad or at a cash machine.",
        "shorts": ["002135", "002148", "002137", "002133", "002134", "002125", "002150", "002153", "002142", "002121", "000552"],
        "reasoning": "Set-new-PIN actions, including 'help me set up a new PIN'.",
    },
    {
        "name": "PIN blocked after too many attempts",
        "description": "Customer's PIN has been blocked because of too many incorrect entries and they want to unblock it.",
        "shorts": ["000533", "000546", "000531", "000534", "000520"],
        "reasoning": "PIN-lockout recovery — includes 'get my PIN unlocked' which is the same action.",
    },

    # ---- PASSCODE / APP LOGIN ----
    {
        "name": "Reset app passcode or password",
        "description": "Customer forgot or wants to reset their app passcode or password to access the account.",
        "shorts": ["001550", "001520", "001538", "001548", "001552", "001523", "001525"],
        "reasoning": "Password/passcode reset.",
    },
    {
        "name": "Account blocked or app login problem",
        "description": "Customer's account is blocked or the app does not recognize them, preventing login.",
        "shorts": ["000542", "000543", "001236", "001213", "001233"],
        "reasoning": "Login/account-blocked — distinct from passcode-reset because the user can't even get to the reset flow.",
    },

    # ---- IDENTITY VERIFICATION ----
    {
        "name": "Identity verification — how to and process",
        "description": "Customer asks how to verify their identity, where, what documents are needed, how long it takes, or what other methods exist.",
        "shorts": ["001192", "001225", "003003", "003017", "003020", "003025", "003036", "001230", "001146"],
        "reasoning": "How-do-I-verify questions and process explanations.",
    },
    {
        "name": "Identity verification failing",
        "description": "Customer reports they cannot verify their identity, the verification is not working, or they lack required documents.",
        "shorts": ["001234", "001238", "001239", "001201", "001205"],
        "reasoning": "Verification failure — distinct from process questions.",
    },
    {
        "name": "Refuses or questions need for identity verification",
        "description": "Customer does not want to verify their identity, or asks why verification is required and pushes back on the data collection.",
        "shorts": ["001189", "001178", "001179", "001185", "001175", "001186", "001187"],
        "reasoning": "Refusal/objection to KYC — distinct intent from failure.",
    },

    # ---- PERSONAL DETAILS ----
    {
        "name": "Update personal details (name, address)",
        "description": "Customer wants to update their personal information — name change, address change, residence move, or general details update.",
        "shorts": ["001154", "001128", "001127", "001150", "001140", "001125"],
        "reasoning": "Personal-details edit — merged because the workflow is identical.",
    },

    # ---- ACCOUNT LIFECYCLE ----
    {
        "name": "Close or delete account",
        "description": "Customer wants to close, delete, or cancel their account, often with a complaint.",
        "shorts": ["001913", "001912", "001887", "001909"],
        "reasoning": "Account-closure intent.",
    },
    {
        "name": "Age requirement for account",
        "description": "Customer asks about the minimum age requirement to open an account.",
        "shorts": ["000511", "000509", "000507", "000480", "000487", "000490"],
        "reasoning": "Age-eligibility questions.",
    },

    # ---- TOP-UP ----
    {
        "name": "How to top up — methods overview",
        "description": "Customer asks generally how to top up their account, what methods are supported, or who can top up.",
        "shorts": ["001332", "002466", "000881", "001354"],
        "reasoning": "General top-up-method discovery.",
    },
    {
        "name": "Top up by bank or card transfer",
        "description": "Customer wants to top up via bank transfer or by using a debit/credit card (including questions about whether that's allowed).",
        "shorts": ["002359", "002350", "002335", "001320", "002829", "001326", "002451"],
        "reasoning": "Top-up-by-transfer / by-card methods, merged per angle. Includes typo 'top up using my car' (presumably card) and 'how do I pay by check' as funding-method questions.",
    },
    {
        "name": "Top up by cash deposit",
        "description": "Customer asks whether and where they can top up using cash, or how to find cash-deposit top-up locations.",
        "shorts": ["002442", "002443", "002444", "002458", "002450", "002468", "002470"],
        "reasoning": "Cash-deposit top-up intent.",
    },
    {
        "name": "Top up by cheque",
        "description": "Customer asks whether they can top up their account by depositing a cheque, and how.",
        "shorts": ["002465", "002473"],
        "reasoning": "Cheque-top-up intent.",
    },
    {
        "name": "Top up using Apple Pay / Google Pay",
        "description": "Customer asks about topping up via mobile-wallet services like Apple Pay or Google Pay.",
        "shorts": ["002989", "002980", "002967", "002969", "002999", "002990"],
        "reasoning": "Wallet-top-up intent.",
    },
    {
        "name": "Auto top-up — set up and limits",
        "description": "Customer asks about auto top-up: enabling it, scheduling it on travel days, where to find the feature, and any limits.",
        "shorts": ["000329", "000328", "000337", "000334", "000346", "000356", "000341", "000330", "000333", "000342", "000323"],
        "reasoning": "Auto-top-up family, including limits, per angle (lean toward grouping).",
    },
    {
        "name": "Top-up limits",
        "description": "Customer asks whether there is a maximum, minimum, or other limit on top-ups.",
        "shorts": ["000738", "000753", "000757", "000730", "000735", "000745", "000746"],
        "reasoning": "Top-up-limit intent — distinct because the question is purely about amounts.",
    },
    {
        "name": "Top-up fees",
        "description": "Customer asks whether topping up incurs a fee, including for specific funding sources (card, transfer, international card).",
        "shorts": ["000618", "002825", "002816", "002817", "000619", "000607", "002801"],
        "reasoning": "Top-up-fee questions.",
    },
    {
        "name": "Top-up failed or reverted",
        "description": "Customer reports their top up failed, didn't go through, was reverted by the app, or was declined.",
        "shorts": ["001026", "001027", "002649", "002665", "002677", "002652", "002659", "002658", "002668", "002641", "002966", "002991", "002978", "002972"],
        "reasoning": "Top-up failure — including Apple-Pay-with-Amex specifics, per angle.",
    },
    {
        "name": "Top-up pending",
        "description": "Customer reports their top-up is stuck showing as pending and not completing.",
        "shorts": ["000648", "000663", "000655", "000678", "000645", "000670", "000671", "000658"],
        "reasoning": "Top-up-pending status complaint.",
    },
    {
        "name": "Top-up card verification code",
        "description": "Customer asks about or cannot find the verification code for their top-up card.",
        "shorts": ["002389", "002387", "002395", "002397", "002374"],
        "reasoning": "Top-up-card-verification subintent.",
    },
    {
        "name": "Currencies accepted for top-up",
        "description": "Customer asks which currencies they can use when adding money to the account.",
        "shorts": ["000894", "000903", "000913", "000884", "000901", "000908"],
        "reasoning": "Currencies-for-topup intent — includes generic 'which forms of currency are accepted' and 'what payment options do I have' since these are typically asked at add-money/funding time in banking77.",
    },

    # ---- TRANSFERS (sending money) ----
    {
        "name": "How to make a transfer",
        "description": "Customer asks generally how to transfer money, including using a credit card to transfer, or describes confusion about the transfer process.",
        "shorts": ["002324", "001352", "001335", "001345", "001328", "002357", "002331", "002333", "002327"],
        "reasoning": "Transfer-howto / initiate intent — 'should I transfer funds, account is out of money' is a transfer-initiation deliberation.",
    },
    {
        "name": "Transfer fee and cost",
        "description": "Customer asks about fees or costs for transfers (sending or receiving), including SEPA / SWIFT specifics and pricing transparency.",
        "shorts": ["000601", "000628", "000632", "000637", "000638", "000620", "002224", "002229", "002238", "002223", "002203", "001875"],
        "reasoning": "Transfer-cost family, including 'why was I charged for transfer'.",
    },
    {
        "name": "International transfer timing / SWIFT / SEPA support",
        "description": "Customer asks how long an international transfer (Europe, US, China, SWIFT) will take or whether SWIFT/SEPA is supported.",
        "shorts": ["000612", "000609", "002067", "002068", "002076", "002057", "002041", "002042", "002062"],
        "reasoning": "Cross-border transfer timing + protocol support — grouped per angle.",
    },
    {
        "name": "Transfer failed or declined",
        "description": "Customer reports a transfer failed, was declined, blocked, or refuses to go through — including beneficiary-denied messages.",
        "shorts": ["001722", "001745", "001747", "001746", "001733", "001732", "001743", "002188", "002164", "002170", "002193", "002296", "002297", "002299", "002309", "002312", "002160", "002181", "002287", "002292"],
        "reasoning": "Broad transfer-failure family per angle; covers 'declined', 'failed', 'blocked', 'can't go through', 'beneficiary denied'.",
    },
    {
        "name": "Transfer pending or not received",
        "description": "Customer reports a transfer is still pending, has not arrived for the recipient, or balance has not updated after a transfer.",
        "shorts": ["001845", "001849", "001855", "001877", "001878", "001879", "002704", "002689", "002684", "002693", "002712", "002713", "000842", "000844", "000848", "000851", "000862", "000874", "002214"],
        "reasoning": "Transfer-pending / not-arrived complaints. Includes 'beneficiary received less than I sent' as a transfer-delivery problem.",
    },
    {
        "name": "Cancel transfer or transaction",
        "description": "Customer wants to cancel a pending transfer or transaction.",
        "shorts": ["000698", "000693", "000682", "000716"],
        "reasoning": "Cancel-action intent.",
    },
    {
        "name": "Receiving money — methods and timing",
        "description": "Customer asks how others can send them money or whether/how a deactivated account will receive incoming payments.",
        "shorts": ["002257", "000246"],
        "reasoning": "Receive-money inbound intent.",
    },

    # ---- CARD PAYMENT TIMING ----
    {
        "name": "Card payment pending",
        "description": "Customer asks why a card purchase is showing as pending or how long that pending state lasts.",
        "shorts": ["001633", "001602", "001613", "001617", "001624", "001630", "000192", "000227", "000204", "000237", "000172"],
        "reasoning": "Pending card-payment status questions, including ATM-pending timing.",
    },

    # ---- DUPLICATE / DOUBLE CHARGES ----
    {
        "name": "Charged twice / duplicate transaction",
        "description": "Customer reports they were charged multiple times for the same transaction or want one of the duplicate charges refunded.",
        "shorts": ["001990", "001992", "001962", "001970", "001971", "001973", "001975", "001995", "001998"],
        "reasoning": "Duplicate-charge intent.",
    },

    # ---- REFUNDS ----
    {
        "name": "Request or check status of refund",
        "description": "Customer wants to request a refund, asks how to get one, or asks where to see a refund that should have processed.",
        "shorts": ["000165", "001778", "001697", "001717", "001786", "001766", "001700", "001684", "001783", "001713", "001710", "001711", "001698"],
        "reasoning": "Refund family — initiate + status + missing refund, merged per angle.",
    },

    # ---- FEES (NON-TRANSFER, NON-TOPUP) ----
    {
        "name": "Card-payment fees and unexpected charges",
        "description": "Customer asks why a fee or extra charge appeared on a card payment or statement, including small unexplained 1-unit charges.",
        "shorts": ["000837", "000813", "000816", "000826", "000827", "000810", "000811", "000832", "000833", "002213", "000185", "000187", "000173", "000168", "000161", "000197", "000191", "000166", "001104"],
        "reasoning": "Unexpected-charge / card-fee family per angle.",
    },
    {
        "name": "ATM / cash-withdrawal fees",
        "description": "Customer asks why they were charged a fee for a cash withdrawal at an ATM, whether withdrawals carry fees, or disputes such a fee.",
        "shorts": ["002882", "002888", "002889", "002892", "002894", "002893", "002895", "002898", "002900", "002907", "002909", "002910", "002916", "002575", "002597", "002880"],
        "reasoning": "Cash-withdrawal-fee family.",
    },

    # ---- EXCHANGE RATES & CURRENCY EXCHANGE ----
    {
        "name": "Currency exchange — how to and supported currencies",
        "description": "Customer asks how to exchange currencies, which fiat currencies are supported, or expresses intent to make a currency exchange.",
        "shorts": ["000404", "000400", "000407", "000412", "000413", "000415", "000255", "000250", "000254", "000257", "000261", "000269", "000271", "000401", "000410", "000428", "000429", "000431", "000268", "002168"],
        "reasoning": "Exchange-how-to and supported-currencies, grouped per angle.",
    },
    {
        "name": "Currency exchange fees and discounts",
        "description": "Customer asks whether currency exchange costs extra, what the exchange fee is, or if there are discounts for frequent exchangers.",
        "shorts": ["002780", "002782", "002783", "002785", "002787", "002790", "002793", "002799", "002760", "002769", "002773", "002776"],
        "reasoning": "Exchange-fee subintent.",
    },
    {
        "name": "Exchange rate source and timing",
        "description": "Customer asks how exchange rates are calculated, where they come from, or whether now is a good time to exchange.",
        "shorts": ["000088", "000089", "000082", "000083", "000093", "000109", "000113"],
        "reasoning": "Exchange-rate-explanation intent.",
    },
    {
        "name": "Wrong exchange rate on purchase",
        "description": "Customer reports they were charged the wrong exchange rate on a card purchase made in a foreign currency.",
        "shorts": ["000135", "000132", "000137", "000145", "000146", "000147", "000149", "000154", "000158", "000159", "000144", "000129"],
        "reasoning": "Wrong-rate-on-payment family.",
    },
    {
        "name": "Wrong exchange rate on cash withdrawal",
        "description": "Customer reports the wrong exchange rate was applied on an ATM cash withdrawal in a foreign currency.",
        "shorts": ["002561", "002567", "002570", "002581", "002593", "002595"],
        "reasoning": "Wrong-rate-on-cash-withdrawal — kept separate because the surface form ('at the ATM') consistently differs.",
    },

    # ---- SALARY ----
    {
        "name": "Receive salary in account / currency configuration",
        "description": "Customer asks whether they can receive their salary in this account and how to configure the deposit currency.",
        "shorts": ["002263", "002247", "002241", "002242", "002246", "002249", "002279"],
        "reasoning": "Salary-deposit family.",
    },

    # ---- SOURCE OF FUNDS ----
    {
        "name": "Source of funds — explain or provide",
        "description": "Customer asks how to look up or report the source of incoming funds, or asks why this information is required.",
        "shorts": ["002021", "002017", "002001", "002004", "002025", "002036", "002038", "002008", "002014", "002029"],
        "reasoning": "Source-of-funds explanation/lookup intent.",
    },

    # ---- CASH / CHECK DEPOSIT (incoming to balance) ----
    {
        "name": "Cash or cheque deposit not showing in balance",
        "description": "Customer reports a cash or cheque deposit they made has not appeared in their balance.",
        "shorts": ["001060", "001066", "001065", "001064", "001063", "001061", "001050", "001052", "001068", "001042", "001073"],
        "reasoning": "Deposit-not-credited family.",
    },
]

# Validate
all_assigned = []
for c in CLUSTERS:
    c["text_ids"] = ids(*c["shorts"])
    all_assigned.extend(c["text_ids"])
    del c["shorts"]

assigned_set = set(all_assigned)
missing = id_set - assigned_set
duplicates = [x for x in all_assigned if all_assigned.count(x) > 1]

if duplicates:
    print(f"DUPLICATES: {sorted(set(duplicates))}", file=sys.stderr)
if missing:
    print(f"MISSING ({len(missing)}):", file=sys.stderr)
    by_id = {r["id"]: r["text"] for r in sample}
    for m in sorted(missing):
        print(f"  {m}: {by_id[m]}", file=sys.stderr)

if duplicates or missing or len(all_assigned) != 600:
    print(f"\nTotals: assigned={len(all_assigned)} unique={len(assigned_set)} expected=600", file=sys.stderr)
    sys.exit(1)

ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = uuid.uuid4().hex[:8]
out_path = OUT_DIR / f"prop_{ts}_{suffix}.json"

proposal = {
    "timestamp": dt.datetime.now().isoformat(),
    "sample_size": 600,
    "sample_strategy": "shared",
    "style": "broad-action",
    "existing_clusters_considered": False,
    "clusters": CLUSTERS,
    "unclustered_ids": [],
    "observations": (
        "Targeted ~65 clusters at the low end of [62,92]. Lean-toward-grouping angle applied: "
        "merged late-card and not-arrived into one delivery-status cluster; combined send-money method (bank transfer + card transfer) "
        "into a single top-up subintent; combined Apple Pay and Google Pay top-up into one wallet bucket; merged refund-request and "
        "refund-status-check; merged personal-details edits (name/address/residence) into one update intent. "
        "Recurring banking77 themes hold: card lifecycle (request/deliver/activate/replace), card failures by surface (ATM vs POS vs contactless vs virtual), "
        "PIN management (find/change/blocked), top-up by funding source, transfers (initiate/fail/pending/cancel/cost), and disputes (duplicate / unauthorized / wrong exchange rate). "
        "Two distinctions I kept fine: unauthorized cash withdrawal vs other unauthorized charges (the surface form 'cash withdrawal I didn't do' is consistent and frequent), "
        "and ATM-cash-withdrawal fees vs card-payment fees (distinct enough that merging would mask a real banking77 intent boundary)."
    ),
}

with out_path.open("w", encoding="utf-8") as f:
    json.dump(proposal, f, indent=2, ensure_ascii=False)

print(f"Wrote {out_path}")
print(f"Cluster count: {len(CLUSTERS)}")
print(f"Assigned: {len(all_assigned)} / 600 (unique: {len(assigned_set)})")
