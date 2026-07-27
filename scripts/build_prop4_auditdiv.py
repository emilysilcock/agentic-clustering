"""Build proposer #4 (fine-grained, ~85 clusters) for banking77 auditdiv sweep.

Reads the shared sample, assigns each id to one of ~85 fine-grained intents,
and writes a single proposal JSON. Designed so every id is in exactly one
real cluster (no leftover bucket).
"""
from __future__ import annotations
import json
import sys
import uuid
import datetime as dt
from pathlib import Path

WS = Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\banking77\seed=0_auditdiv")
SAMPLE = WS / "samples" / "shared_proposer_sample.json"
OUT_DIR = WS / "proposals"

sys.stdout.reconfigure(encoding="utf-8")

with SAMPLE.open(encoding="utf-8") as f:
    sample = json.load(f)

# id -> text lookup, all ids set
id_to_text = {row["id"]: row["text"] for row in sample}
all_ids = set(id_to_text)

# ---------------------------------------------------------------------------
# Cluster definitions. Each is (name, description, list_of_ids).
# Designed for fine-grained intents (~85 clusters). Splits applied:
#   - status check vs action request (e.g. "where is my refund" vs "I need a refund")
#   - failed attempt vs general info (e.g. "transfer declined" vs "how do transfers work")
#   - lost vs stolen vs lost-phone (separate intents in banking77)
#   - delivery time vs tracking vs delivery location
#   - top-up by source (cheque/cash/bank-transfer/apple-pay/google-pay/card/AmEx) kept separate
# ---------------------------------------------------------------------------

clusters: list[tuple[str, str, list[str]]] = []

# 1. ATM withdrawal declined / cash not dispensed by ATM
clusters.append((
    "ATM cash not dispensed / withdrawal not given",
    "User attempted an ATM withdrawal and got no cash or the machine refused to dispense; about the machine itself failing to give money rather than card decline.",
    [
        "banking77-test-001591",  # ATM isn't giving out any money
        "banking77-test-001589",  # Why didn't the ATM give me my money?
        "banking77-test-001573",  # can't get money out of the ATM
        "banking77-test-001583",  # can't take money out of the ATM
    ],
))

# 2. ATM withdrawal declined — card refused at ATM
clusters.append((
    "ATM card declined / withdrawal refused",
    "User was specifically declined at an ATM when trying to withdraw cash; the card or transaction was rejected.",
    [
        "banking77-test-001577",  # Why was I declined at the ATM today
        "banking77-test-001576",  # declined when I tried to get cash
        "banking77-test-001565",  # ATM keeps declining my card
        "banking77-test-001571",  # card declining by ATM
        "banking77-test-001570",  # card withdrawal was declined this morning
    ],
))

# 3. ATM gave wrong (smaller) cash amount
clusters.append((
    "ATM dispensed wrong cash amount",
    "User received a different (typically smaller) amount of cash than they requested at the ATM, but the app may still show the full charge.",
    [
        "banking77-test-000782",  # asked for more than I got
        "banking77-test-000764",  # got less cash than specified
        "banking77-test-000779",  # tried to get $100 but got $20
        "banking77-test-000786",  # asked for 100 but only got 20
        "banking77-test-000787",  # didn't receive the right amount of cash back
        "banking77-test-000797",  # didn't get the right amount at the ATM
        "banking77-test-000798",  # did not receive correct cash upon withdrawal
        "banking77-test-000780",  # not received the amount of cash I was supposed to
        "banking77-test-000770",  # wrong amount of cash from ATM, app shows much more
        "banking77-test-000772",  # ATM gave wrong amount, app shows charged
        "banking77-test-000775",  # cash withdrawal partly declined
        "banking77-test-002755",  # Cash I didn't get shows in my app
        "banking77-test-002724",  # app shows some cash I didn't get
        "banking77-test-002759",  # app made a mistake and said I made a cash withdrawal
    ],
))

# 4. ATM swallowed / trapped card
clusters.append((
    "ATM swallowed / trapped my card",
    "User's physical card was eaten or retained by an ATM machine and they need help getting it back.",
    [
        "banking77-test-001952",  # card swallowed by ATM
        "banking77-test-001933",  # ATM swallowed card, how to get back
        "banking77-test-001953",  # taking out funds, unable to regain card
        "banking77-test-001951",  # card trapped in ATM
        "banking77-test-001948",  # retrieve trapped card
        "banking77-test-001926",  # get ATM card back from machine
        "banking77-test-001928",  # ATM didn't give me the card back
        "banking77-test-001931",  # how can I get my card back out
        "banking77-test-001941",  # what happens if ATM doesn't give card back
        "banking77-test-001958",  # disappeared in machine
    ],
))

# 5. ATM withdrawal — fee charged / questioning fee
clusters.append((
    "Charged a fee for ATM cash withdrawal",
    "User notices or questions a fee on a recent cash withdrawal from an ATM.",
    [
        "banking77-test-002888",  # extra charge for money withdrawn
        "banking77-test-002909",  # thought cash withdrawals were free
        "banking77-test-002882",  # why charged when withdrawing cash
        "banking77-test-002889",  # charged a fee for withdrawing cash
        "banking77-test-002894",  # fee charged for recent withdrawal
        "banking77-test-002895",  # got a fee for ATM withdrawal
        "banking77-test-002892",  # had to pay a fee when I got cash
        "banking77-test-002893",  # noticed fee for withdrawal, didn't know fees were charged
        "banking77-test-002898",  # is there a fee for withdrawing cash
        "banking77-test-002900",  # account charged for using ATM
        "banking77-test-002907",  # charged extra for withdrawal
        "banking77-test-002910",  # more fees for withdrawing my own cash
        "banking77-test-002916",  # wrong fee charged at this ATM
        "banking77-test-002597",  # outrageous charges at ATM
        "banking77-test-002575",  # additional costs for withdrawal in GBP
        "banking77-test-002880",  # see some fees for cash withdraw
    ],
))

# 6. ATM withdrawal — wrong exchange rate
clusters.append((
    "Wrong exchange rate on ATM cash withdrawal",
    "User used a foreign ATM and the exchange rate applied to the cash withdrawal looks wrong.",
    [
        "banking77-test-002561",  # ATM exchanged wrong amount from another currency
        "banking77-test-002595",  # ATM in foreign currency, rate is wrong
        "banking77-test-002593",  # rate of exchange isn't right for cash withdrawal
        "banking77-test-002570",  # exchange rate given for cash withdrawal is wrong
        "banking77-test-002567",  # foreign currency at ATM, rate inaccurate
    ],
))

# 7. ATM locator / where can I withdraw
clusters.append((
    "Find / locate an ATM nearby",
    "User wants to find an ATM, especially the nearest one or one that accepts their card.",
    [
        "banking77-test-001463",  # any ATMs near me
        "banking77-test-001442",  # closest Mastercard ATM
        "banking77-test-001462",  # finding nearest ATM
    ],
))

# 8. Which ATMs accept my card
clusters.append((
    "Which ATMs accept my card",
    "User asks which ATM networks or specific ATMs will accept their card for withdrawal.",
    [
        "banking77-test-001466",  # Where can I use this card at an ATM
        "banking77-test-001449",  # Which ATMs accept this placard
        "banking77-test-001468",  # What ATMs can I withdraw money from
        "banking77-test-001471",  # ATMs that will accept this card
        "banking77-test-001444",  # specific ATMs this card can be used at
        "banking77-test-001478",  # places where I can't withdraw money
    ],
))

# 9. ATM withdrawal — how long to post
clusters.append((
    "How long does an ATM withdrawal take to post",
    "User wants to know how long a cash withdrawal takes to show up / clear on their statement.",
    [
        "banking77-test-000237",  # how long does it take to post atm withdrawal
        "banking77-test-000204",  # how long does cash withdrawal take to pend
    ],
))

# 10. Unauthorized / suspicious cash withdrawal
clusters.append((
    "Unauthorized or unrecognized cash withdrawal",
    "User sees a cash withdrawal in their account that they did not make or do not recognize.",
    [
        "banking77-test-002736",  # cash withdrawal that's not mine
        "banking77-test-002726",  # suspicious cash withdrawal on account
        "banking77-test-002758",  # cash withdrawal showing up that I didn't do
        "banking77-test-002732",  # extra cash withdrawal I didn't authorize
        "banking77-test-002737",  # weird withdrawal
        "banking77-test-002744",  # odd withdrawal on account
        "banking77-test-002725",  # random withdrawal in app
        "banking77-test-002739",  # someone else withdrew cash from my account
        "banking77-test-002756",  # somebody removed money in a town I haven't been to, freeze
    ],
))

# 11. Card payment declined / not working at merchant
clusters.append((
    "Card payment declined at merchant",
    "User tried to pay with their physical card at a merchant and it was declined or did not work, despite having funds.",
    [
        "banking77-test-001822",  # debit card declined when I have money
        "banking77-test-001813",  # card payment hasn't worked, I have funds
        "banking77-test-001811",  # new card keeps getting declined
        "banking77-test-001828",  # payments declining again and again
        "banking77-test-001837",  # tried using card, kept getting declined
        "banking77-test-001835",  # card didn't work in shops
        "banking77-test-001834",  # card payment did not complete
        "banking77-test-001824",  # no idea why card payment did not work
        "banking77-test-002544",  # card just not working
        "banking77-test-002094",  # why did app refuse to make approved payment
    ],
))

# 12. Card payment pending (status)
clusters.append((
    "Card payment is still pending",
    "User made a card payment and it is stuck in pending status on their statement.",
    [
        "banking77-test-001633",  # bought things, payment is pending
        "banking77-test-001602",  # card payment still pending
        "banking77-test-001613",  # bought stuff, payment pending
        "banking77-test-001617",  # purchases from this morning are still pending
        "banking77-test-001624",  # how long will this card payment stay pending
        "banking77-test-001630",  # why are my payments pending
        "banking77-test-000227",  # unable to take money but statement says pending
    ],
))

# 13. Card payment was cancelled / reversed
clusters.append((
    "Card payment was cancelled or reverted",
    "User's card payment was cancelled, stopped, or reverted unexpectedly.",
    [
        "banking77-test-002083",  # card payment was cancelled
        "banking77-test-002080",  # card payment has been stopped
        "banking77-test-002115",  # canceled payment for my card
    ],
))

# 14. Card payment fee question
clusters.append((
    "Why was I charged a fee on a card payment",
    "User questions why there is a fee tied to a specific card payment, or asks about card-payment fees in general.",
    [
        "banking77-test-000837",  # charged fee for card payment
        "banking77-test-000816",  # made card payment and fee showing up
        "banking77-test-000826",  # fee when you pay with card
        "banking77-test-000827",  # charged with fee for paying with card
        "banking77-test-000832",  # fees when using a card
        "banking77-test-000833",  # why do you charge for payments
        "banking77-test-000810",  # why did I pay extra because I paid with card
        "banking77-test-000811",  # why are there fees for card usage
        "banking77-test-000813",  # fee charged with this card payment
    ],
))

# 15. Virtual card not working / declined
clusters.append((
    "Virtual card declined or not working",
    "Issues using a virtual (online) card: declined for billing, rejected by merchant, or generally not functioning.",
    [
        "banking77-test-002532",  # virtual card declined for automatic billing
        "banking77-test-002541",  # what to do if virtual card won't work
        "banking77-test-002543",  # can't get virtual card to work
        "banking77-test-002553",  # disposable virtual card rejected by merchant
    ],
))

# 16. Contactless / NFC payment problem
clusters.append((
    "Contactless / NFC payment not working",
    "Tap-to-pay / contactless / NFC payments not working, often on transit or at shops.",
    [
        "banking77-test-000570",  # contactless has stopped working
        "banking77-test-000572",  # Contactless isn't working
        "banking77-test-000583",  # NFC payment wouldn't work on the bus
        "banking77-test-000577",  # how to make contactless work for the metro
        "banking77-test-000567",  # why my contactless won't work
    ],
))

# 17. Card not working — generic / vague diagnosis
clusters.append((
    "Card not working — diagnose the problem",
    "User says their card is broken or not working without specifying ATM/merchant; asks the bank to look into it.",
    [
        "banking77-test-000388",  # card not working
        "banking77-test-000382",  # I broke my card
        "banking77-test-000383",  # card stopped working
        "banking77-test-000374",  # I think my card is broken
        "banking77-test-000375",  # why isn't my card working
        "banking77-test-000373",  # can a card stop working
        "banking77-test-000371",  # explaining why card is not working
        "banking77-test-000367",  # identify the problem with my bank card
        "banking77-test-000599",  # should I reinstall the payment app (generic troubleshooting)
    ],
))

# 18. Reactivate found card
clusters.append((
    "Reactivate a previously lost/frozen card",
    "User had previously reported card lost or had it frozen, now found it, wants to reactivate.",
    [
        "banking77-test-000076",  # reactivate card, found in jacket
        "banking77-test-000062",  # found my card, reactivate it
    ],
))

# 19. Activate a new card
clusters.append((
    "How do I activate my new card",
    "User wants to activate a newly issued card so they can start using it.",
    [
        "banking77-test-002864",  # activating my card
        "banking77-test-002868",  # how do I activate my new card
        "banking77-test-002857",  # please activate my card
        "banking77-test-002871",  # process for activating card
        "banking77-test-002872",  # how do I use my new card
        "banking77-test-002879",  # what do I need to do to activate
        "banking77-test-002844",  # activate so I can start using
    ],
))

# 20. Link / register a card to the app
clusters.append((
    "Link or register a card to the app",
    "User wants to link a new (often external) card to their account in the app.",
    [
        "banking77-test-000054",  # link new card
        "banking77-test-000050",  # how do I link a new card
        "banking77-test-000058",  # where on website to link card
        "banking77-test-000060",  # link credit card with you
        "banking77-test-000057",  # view card I received in the app
        "banking77-test-000066",  # app doesn't show the card I received
        "banking77-test-000000",  # how do I locate my card  (in app)
    ],
))

# 21. Card expiring — what to do
clusters.append((
    "Card is about to expire — next steps",
    "User's existing card is expiring and they want to know what to do / replacement process.",
    [
        "banking77-test-002926",  # card about to expire, what do I need
        "banking77-test-002925",  # card's expiring, what happens
        "banking77-test-002923",  # what do I do when my card expires
        "banking77-test-002943",  # order new card before current expires
        "banking77-test-002947",  # order new card before it expires
    ],
))

# 21b. Order replacement card — cost & time
clusters.append((
    "Order new/replacement card — cost and delivery time",
    "User wants combined info on cost and delivery time for getting a new or replacement card, often because the current is expiring.",
    [
        "banking77-test-002415",  # procedure for getting another card
        "banking77-test-002927",  # how long and how much for new card
        "banking77-test-002931",  # card expiring, cost and time for new
        "banking77-test-002946",  # order new card, cost and time
        "banking77-test-002959",  # cost and when receive new card replacement
    ],
))

# 22. Card not arrived yet — waiting for delivery
clusters.append((
    "Card has not arrived yet — status of delivery",
    "User is waiting on a card that was ordered or shipped and hasn't arrived.",
    [
        "banking77-test-000004",  # card has not arrived yet
        "banking77-test-000013",  # still waiting on new card
        "banking77-test-000036",  # think my card is lost because hasn't arrived
        "banking77-test-000009",  # is it normal to wait over a week
        "banking77-test-000312",  # waiting for card to arrive
    ],
))

# 23. Card delivery time question
clusters.append((
    "How long until my card is delivered",
    "User asks how long card delivery will take, including to specific countries.",
    [
        "banking77-test-000302",  # how long for card to arrive
        "banking77-test-000282",  # need my card quick
        "banking77-test-000286",  # how long to deliver to US
        "banking77-test-000290",  # delivery to US take long
        "banking77-test-000294",  # delivered by a specific date
        "banking77-test-000303",  # delivery time to United States
        "banking77-test-000032",  # how do I know when card will arrive
    ],
))

# 24. Card delivery location / where can it be delivered
clusters.append((
    "Where can my card be delivered / shipping locations",
    "User asks about which locations / addresses / countries the bank will ship a physical card to.",
    [
        "banking77-test-002506",  # where can my card be delivered
        "banking77-test-002509",  # locations a card can be delivered
        "banking77-test-002491",  # do you ship cards to where I live
        "banking77-test-002483",  # where can I get my card at
        "banking77-test-002480",  # how do I receive my physical card
    ],
))

# 25. Card tracking number request
clusters.append((
    "Tracking number for shipped card",
    "User wants the tracking / shipment tracking number for a card that was sent to them.",
    [
        "banking77-test-000006",  # tracking number for new card
        "banking77-test-000039",  # where is the tracking number
    ],
))

# 26. Order a physical card (general request)
clusters.append((
    "Request / order a physical card",
    "User wants to request, order, or get a physical (non-virtual) card; general inquiries about getting one.",
    [
        "banking77-test-002502",  # get a physical card through this app
        "banking77-test-002488",  # request a physical card
        "banking77-test-002497",  # tell me how to get a physical card
        "banking77-test-002499",  # non-electronic card available
    ],
))

# 27. Cost / fees for physical card
clusters.append((
    "Fees for a physical card",
    "User asks about the cost or fees specifically for getting a physical card.",
    [
        "banking77-test-002505",  # fees to get a physical card
        "banking77-test-002507",  # what will I be charged for a physical card
    ],
))

# 28. Extra / additional cards (multiple cards on one account)
clusters.append((
    "Additional or multiple physical cards on one account",
    "User wants to add additional physical cards (for self or family) to their account, or asks about per-account card limits/fees.",
    [
        "banking77-test-002406",  # receive more physical cards
        "banking77-test-002417",  # daughter wants a card too
        "banking77-test-002404",  # fee for extra cards
        "banking77-test-002435",  # extra amount to be paid for more than one card
        "banking77-test-002401",  # how many cards for one account
    ],
))

# 29. Virtual / disposable card — what is it / how to get
clusters.append((
    "Virtual / disposable card — explainer & how to get one",
    "User asks what a virtual or disposable card is, how it works, or how to obtain one.",
    [
        "banking77-test-002623",  # process of obtaining disposable virtual card
        "banking77-test-002626",  # possible to also get a disposable virtual card
        "banking77-test-002600",  # description of how to use disposable virtual card
        "banking77-test-002610",  # explain disposable cards
        "banking77-test-002602",  # tell me about getting a virtual disposable card
        "banking77-test-002616",  # explain disposable cards
        "banking77-test-002622",  # how does a virtual card work
        "banking77-test-002638",  # what are disposable cards
        "banking77-test-000921",  # how do you get a virtual card
        "banking77-test-000956",  # reorder my virtual card
    ],
))

# 30. Disposable card limits
clusters.append((
    "Disposable card usage / quantity limits",
    "User asks about limits on disposable cards: how many they can have, how many uses, daily caps.",
    [
        "banking77-test-001364",  # can I have more than one disposable card
        "banking77-test-001362",  # disposable card multiple times a day cutoff
        "banking77-test-001373",  # five payments on each disposable
        "banking77-test-001369",  # amount of disposable cards each day
        "banking77-test-001365",  # disposable cards limit
        "banking77-test-001372",  # need multiple disposable cards
        "banking77-test-001368",  # how many disposable cards
        "banking77-test-001380",  # restrictions for disposable cards
        "banking77-test-001382",  # several disposable cards in a day
        "banking77-test-001383",  # how many disposable cards in single day
        "banking77-test-001385",  # limits on disposable cards
        "banking77-test-001393",  # how many times can I use a disposable card
        "banking77-test-001375",  # max transactions on one card
    ],
))

# 31. Lost card — report / replace
clusters.append((
    "Lost card — report and replace",
    "User has lost their physical card and needs to report it / get a replacement.",
    [
        "banking77-test-000446",  # Oh no! I lost my card!
        "banking77-test-000445",  # I lost my card
        "banking77-test-000471",  # card is lost, what do I do
        "banking77-test-000470",  # how to report card lost or stolen (generic phrasing)
    ],
))

# 32. Stolen card — report / freeze / replace
clusters.append((
    "Card stolen — report and freeze/replace",
    "User's card has been stolen and they need to report it stolen, freeze it, or get a replacement.",
    [
        "banking77-test-000479",  # card was stolen
        "banking77-test-000475",  # somebody stolen my card
        "banking77-test-000474",  # report a stolen card
        "banking77-test-000472",  # in Spain, stuff stolen, need new card and old frozen
        "banking77-test-000454",  # lost wallet, block card and replace
    ],
))

# 33. (merged) — generic "lost or stolen" phrasing folded into "Lost card — report and replace"


# 34. Card hacked / compromised (someone using it)
clusters.append((
    "Card compromised — unauthorized purchases",
    "User sees charges on their card they didn't make and suspects someone else is using it; wants to freeze / block.",
    [
        "banking77-test-001402",  # someone used my card without permission
        "banking77-test-001407",  # freeze my card because someone used it
        "banking77-test-001419",  # random purchases, was it hacked
        "banking77-test-001434",  # think my child used my card
        "banking77-test-001110",  # charge I don't recall, account compromised
        "banking77-test-001111",  # didn't make this payment
        "banking77-test-001516",  # someone has taken my money
    ],
))

# 35. Freeze card from the app
clusters.append((
    "Freeze card through the app",
    "User wants to freeze their card from within the app, not necessarily due to loss/theft.",
    [
        "banking77-test-001421",  # freeze my card using the app
    ],
))

# 36. Lost phone — secure account
clusters.append((
    "Lost phone — secure account access",
    "User has lost their phone (or had it stolen) and is worried about app access; wants to freeze accounts or secure access.",
    [
        "banking77-test-001658",  # lost phone, freeze accounts
        "banking77-test-001647",  # phone stolen or lost
        "banking77-test-001645",  # can't find my phone
        "banking77-test-001651",  # lost phone, need help securing
        "banking77-test-001662",  # lost phone, don't want others access
        "banking77-test-001671",  # someone stole my phone
        "banking77-test-001678",  # got mugged, can't use app
    ],
))

# 37. PIN — change / update / set new
clusters.append((
    "Change or set a new card PIN",
    "User wants to change, update or set a new PIN on their card.",
    [
        "banking77-test-002135",  # update PIN to new number
        "banking77-test-002148",  # set up a new PIN
        "banking77-test-002137",  # new pin needs to be set
        "banking77-test-002134",  # set a new pin
        "banking77-test-002133",  # new PIN selection
        "banking77-test-002125",  # change my PIN
        "banking77-test-002121",  # where do I change my PIN
        "banking77-test-002142",  # change PIN at cash machine
        "banking77-test-002150",  # change PIN abroad
        "banking77-test-002153",  # update PIN on account
        "banking77-test-000552",  # reset my PIN
        "banking77-test-001275",  # what do I need for PIN
    ],
))

# 38. Don't have / can't find / forgot PIN
clusters.append((
    "Don't have / can't find my PIN",
    "User doesn't have their PIN, can't find it, or asks where to view/look up their PIN.",
    [
        "banking77-test-001242",  # do not have my pin
        "banking77-test-001270",  # where can I find my PIN
        "banking77-test-001261",  # find the card PIN
        "banking77-test-001240",  # what do I do with my card PIN
        "banking77-test-001278",  # what is my card's PIN
        "banking77-test-000521",  # where can I view my PIN
    ],
))

# 39. PIN blocked / locked — unblock
clusters.append((
    "PIN blocked from too many attempts — unblock",
    "User entered the wrong PIN too many times or otherwise got blocked and wants to unblock it.",
    [
        "banking77-test-000533",  # unblock a blocked pin
        "banking77-test-000534",  # ATM won't accept PIN attempts
        "banking77-test-000546",  # wrong pin too many times, blocked
        "banking77-test-000531",  # tried to enter PIN too often
        "banking77-test-000520",  # how do I get my PIN unlocked
    ],
))

# 40. Account blocked / can't log in
clusters.append((
    "Account blocked / locked out — how to log in",
    "User's account itself (not just PIN) is blocked or won't let them log in.",
    [
        "banking77-test-000542",  # account is blocked, how to log in
        "banking77-test-000543",  # why did I get blocked
        "banking77-test-001236",  # app won't let me log in as myself
        "banking77-test-001213",  # not being recognized by the app
        "banking77-test-001233",  # app doesn't think it's me
    ],
))

# 41. Forgot passcode / reset password
clusters.append((
    "Forgot or reset app passcode / password",
    "User forgot their app password / passcode and needs to reset it.",
    [
        "banking77-test-001550",  # reset password
        "banking77-test-001525",  # password not accepted, need to reset
        "banking77-test-001523",  # passcode doesn't seem to work
        "banking77-test-001520",  # forgot passcode
        "banking77-test-001538",  # reset the passcode
        "banking77-test-001548",  # forgot code to access app
        "banking77-test-001552",  # reset the passcode
    ],
))

# 42. Identity verification — how / where / what's needed
clusters.append((
    "Identity verification — how to do it",
    "User asks how / where to verify their identity, what documents or steps are required.",
    [
        "banking77-test-001192",  # other methods to verify identity
        "banking77-test-001225",  # what do I need to verify my id
        "banking77-test-001201",  # don't have what is required
        "banking77-test-003003",  # how can I verify my identity
        "banking77-test-003017",  # steps for identity checks
        "banking77-test-003020",  # how does identity get verified
        "banking77-test-003025",  # where do I verify my identity
        "banking77-test-003036",  # documentation needed for identity check
    ],
))

# 43. Identity verification — failing / can't verify
clusters.append((
    "Trouble verifying identity",
    "User is having problems getting their identity verified; the process is failing.",
    [
        "banking77-test-001234",  # trouble verifying id
        "banking77-test-001238",  # can't verify my id
        "banking77-test-001239",  # problem verifying identity
        "banking77-test-001205",  # impossible to verify identity
        "banking77-test-001230",  # how long for ID to verify
    ],
))

# 44. Identity verification — refusing / objecting
clusters.append((
    "Refusing identity verification / unhappy with KYC",
    "User does not wish to verify identity, objects to data collection, or asks why so much is required.",
    [
        "banking77-test-001189",  # do not wish to verify identity
        "banking77-test-001187",  # won't verify identity
        "banking77-test-001185",  # why require so many identity details
        "banking77-test-001175",  # don't like having to fill out so much identity info
    ],
))

# 45. Can I use account before ID verified
clusters.append((
    "Use account while identity not yet verified",
    "User asks if they can use the account before identity verification has completed.",
    [
        "banking77-test-001178",  # use account even though verification not through
        "banking77-test-001179",  # need to verify identity to use account
        "banking77-test-001186",  # since id not verified, when can I use
    ],
))

# 46. Source of funds explanation / where money came from
clusters.append((
    "Source of funds — view / explain incoming money",
    "User wants to know where money came from on their account or asks the bank to look up source of funds.",
    [
        "banking77-test-002021",  # lookup where funds came from
        "banking77-test-002017",  # check where funds came from
        "banking77-test-002004",  # where do funds come from
        "banking77-test-002025",  # where does my money come from
        "banking77-test-002001",  # info about source of funds
        "banking77-test-002036",  # see source of funds
        "banking77-test-002038",  # where does all this money come from
    ],
))

# 47. Source of funds — why does bank need info / KYC
clusters.append((
    "Why does the bank ask for source of funds info",
    "User asks why the bank needs source-of-funds info or what info is needed.",
    [
        "banking77-test-002008",  # why need all this source of funds info
        "banking77-test-002014",  # what info needed for source of funds
        "banking77-test-002029",  # why need to know where money is coming from
    ],
))

# 48. Personal details change — name
clusters.append((
    "Change my name on the account",
    "User wants to change the name on their account.",
    [
        "banking77-test-001154",  # change my name
        "banking77-test-001150",  # how to change name
        "banking77-test-001140",  # change my name procedure
    ],
))

# 49. Personal details change — address / generic
clusters.append((
    "Change my address or personal details",
    "User wants to update their address or other personal details after moving etc.",
    [
        "banking77-test-001128",  # I moved, update details
        "banking77-test-001127",  # change personal details
        "banking77-test-001146",  # update residence details
        "banking77-test-001125",  # address has changed
    ],
))

# 50. Close / delete account
clusters.append((
    "Close or delete my account",
    "User wants to close, delete, or terminate their account.",
    [
        "banking77-test-001913",  # close my account, don't like service
        "banking77-test-001912",  # delete my account, sucks
        "banking77-test-001887",  # close my account
        "banking77-test-001909",  # want account deleted
    ],
))

# 51. Age / country eligibility — open account
clusters.append((
    "Account eligibility — age and country",
    "User asks about the minimum age or country eligibility for opening an account, or where the service operates.",
    [
        "banking77-test-000511",  # how old to open account
        "banking77-test-000509",  # age requirements for account
        "banking77-test-000490",  # minimum age to open account
        "banking77-test-000507",  # how old to open account
        "banking77-test-000487",  # youngest age to have account
        "banking77-test-000480",  # how old to get account
        "banking77-test-003056",  # is my country supported
        "banking77-test-003058",  # where can I find your locations
        "banking77-test-003069",  # in what countries do you do business
    ],
))

# 52. Get card outside the UK / abroad
clusters.append((
    "Get a card from outside the UK / abroad",
    "User lives outside the UK and asks whether they can get / receive / use a card.",
    [
        "banking77-test-003059",  # get a card living outside UK
        "banking77-test-003045",  # card outside UK
        "banking77-test-003048",  # live in US, how to get a card
        "banking77-test-003054",  # in US, get a new card
        "banking77-test-003061",  # use the card in Europe
    ],
))

# 53. Top up — list of accepted methods / overview
clusters.append((
    "Top up — what methods are available",
    "User asks generally how to top up, or what methods/options the service supports.",
    [
        "banking77-test-001332",  # how can I top up
        "banking77-test-000881",  # support options, top my card
        "banking77-test-002466",  # methods to top up
        "banking77-test-001354",  # who can top up my accounts
    ],
))

# 54. Top up — cheque
clusters.append((
    "Top up by cheque",
    "User asks specifically about topping up via cheque.",
    [
        "banking77-test-002465",  # top up with cheque
        "banking77-test-002442",  # top up with cheque
        "banking77-test-002473",  # top up with cheque
        "banking77-test-002451",  # pay by check
    ],
))

# 55. Top up — cash deposit
clusters.append((
    "Top up by cash deposit",
    "User asks how / where to top up using cash deposit.",
    [
        "banking77-test-002470",  # deposit cash into account
        "banking77-test-002443",  # top up with cash
        "banking77-test-002444",  # find top up by cash deposit
        "banking77-test-002468",  # can't find anywhere to load using cash
        "banking77-test-002450",  # locate topping up with cash
        "banking77-test-002458",  # locations to top up with cash
    ],
))

# 56. Top up — bank transfer (from another account)
clusters.append((
    "Top up via bank transfer from another account",
    "User wants to top up by initiating a bank transfer from another account.",
    [
        "banking77-test-002359",  # bank transfer to refill account
        "banking77-test-002350",  # transfer from accounts to top up, steps
        "banking77-test-002335",  # transfer from bank to top up account
        "banking77-test-002357",  # transfer money into my account
        "banking77-test-002333",  # transfer funds into my account, how
        "banking77-test-002331",  # transfer from other bank account
        "banking77-test-002327",  # transfer funds, account out of money
    ],
))

# 57. Top up — Apple Pay
clusters.append((
    "Top up via Apple Pay",
    "User asks whether/how to top up using Apple Pay.",
    [
        "banking77-test-002980",  # Apple Pay possible top up option
        "banking77-test-002969",  # use Apple Pay to top up
        "banking77-test-002967",  # top up with Apple Pay
        "banking77-test-002990",  # use Apple Pay to put money in account
    ],
))

# 58. Top up — Google Pay / Apple Watch
clusters.append((
    "Top up via Google Pay or other wallet (e.g. Apple Watch)",
    "User asks about topping up via Google Pay, Apple Watch, or similar wearable/wallet beyond plain Apple Pay.",
    [
        "banking77-test-002989",  # top up with Apple Watch
        "banking77-test-002999",  # Google pay and top up
    ],
))

# 59. Top up — card (debit/credit, generic)
clusters.append((
    "Top up using a bank card",
    "User asks about topping up with a generic bank/credit/debit card.",
    [
        "banking77-test-001320",  # top up account with a card
        "banking77-test-002829",  # use bank card to top up
        "banking77-test-001326",  # top up using my car [typo for card]
    ],
))

# 60. Top up — AmEx / specific brand
clusters.append((
    "Top up with American Express specifically",
    "User wants to add money using American Express, often with Apple Pay attached.",
    [
        "banking77-test-000894",  # add money through American Express
    ],
))

# 61. Top up via AmEx + Apple Pay failing
clusters.append((
    "AmEx + Apple Pay top-up failing",
    "User combined American Express through Apple Pay and the top-up is failing.",
    [
        "banking77-test-002991",  # AmEx in Apple Pay, top up failing
        "banking77-test-002978",  # AmEx problem with apple pay top up
        "banking77-test-002972",  # AmEx with Apple Pay problem
    ],
))

# 62. Top up failed / declined generally
clusters.append((
    "Top up failed or was declined",
    "User's top-up attempt failed, was declined, or the app rejected it (excluding pending-status).",
    [
        "banking77-test-002649",  # topup didn't go through
        "banking77-test-002652",  # top up failed
        "banking77-test-002658",  # credit card declined for top up
        "banking77-test-002659",  # topped up but app did not accept
        "banking77-test-002665",  # how did top up fail
        "banking77-test-002668",  # card denied for top up
        "banking77-test-002677",  # top up failed, fix
        "banking77-test-002641",  # tried to top up with card, failed
        "banking77-test-002966",  # top up on Google Pay isn't working
        "banking77-test-001026",  # app reverted my top off
        "banking77-test-001027",  # app reverted my action when topping up
    ],
))

# 63. Top up pending — status
clusters.append((
    "Top up is stuck pending",
    "User's top-up is stuck on pending status and they want to know why / when it will complete.",
    [
        "banking77-test-000648",  # top-up still pending
        "banking77-test-000645",  # top-up is pending
        "banking77-test-000655",  # top-up just says pending
        "banking77-test-000663",  # why is my top up still pending
        "banking77-test-000658",  # something wrong, top up close to 2 hours pending
        "banking77-test-000670",  # top up stuck in pending
        "banking77-test-000671",  # doesn't look like top-up completed
        "banking77-test-000678",  # top-up has a definite problem, pending
    ],
))

# 64. Top up — fees question
clusters.append((
    "Fees on topping up",
    "User asks about fees on top-ups (general, by card, by transfer, with international card).",
    [
        "banking77-test-002801",  # fees for adding money with international card
        "banking77-test-000618",  # fees for top-ups
        "banking77-test-002825",  # does topping up card have fee
        "banking77-test-002816",  # fee included if top up by card
        "banking77-test-002817",  # European bank card for top up, charged
        "banking77-test-000619",  # topping up by transfer charge
        "banking77-test-000607",  # pay for topping up by transfer
    ],
))

# 65. Top up — limit
clusters.append((
    "Top-up amount limit",
    "User asks if there is a maximum amount or any limit on top-ups.",
    [
        "banking77-test-000753",  # amount limit to top-up
        "banking77-test-000738",  # limit to top-ups
        "banking77-test-000757",  # restrictions to top-up
        "banking77-test-000730",  # maximum top up
        "banking77-test-000735",  # limit to top-up
        "banking77-test-000746",  # top-up limit
        "banking77-test-000745",  # maximum amount of top-ups
    ],
))

# 66. Top-up verification code (the physical/top-up card code)
clusters.append((
    "Top-up card verification code",
    "User is looking for or asking about the verification code on/for their top-up card.",
    [
        "banking77-test-002389",  # find the top-up card's verification code
        "banking77-test-002395",  # cannot locate verification code for top-up card
        "banking77-test-002397",  # finding verification code for top-up card
        "banking77-test-002387",  # is there verification code for top-up card
        "banking77-test-002374",  # why does top-up need verification
    ],
))

# 67. Auto top-up — setup / scheduling
clusters.append((
    "Auto top-up — set up scheduled top-ups",
    "User wants to schedule automatic top-ups (intervals, while traveling).",
    [
        "banking77-test-000329",  # add money automatically at intervals while away
        "banking77-test-000328",  # add money at time intervals when I travel
        "banking77-test-000341",  # auto top-up card on certain days while traveling
        "banking77-test-000323",  # running out, can I auto top up
        "banking77-test-000356",  # top-up automatically
        "banking77-test-000346",  # auto top up if low
        "banking77-test-000330",  # find the auto-top feature
        "banking77-test-000333",  # where is theft-top option [auto top]
        "banking77-test-000342",  # auto top-up option location
    ],
))

# 68. Auto top-up — limits
clusters.append((
    "Auto top-up limits",
    "User asks about limits specifically on auto top-up.",
    [
        "banking77-test-000337",  # limits on auto top-up
        "banking77-test-000334",  # limits on auto top-up
    ],
))

# 69. Currency exchange — how to / process
clusters.append((
    "How to exchange currency",
    "User asks how to do a currency exchange / what the process is, or initiates a request to exchange.",
    [
        "banking77-test-000412",  # process for GBP to AUD
        "banking77-test-000415",  # exchange GBP to AUD
        "banking77-test-000255",  # change currencies to euros
        "banking77-test-000413",  # get some money exchanged
        "banking77-test-000250",  # exchange money for EUR
        "banking77-test-000268",  # currency exchange to EU
        "banking77-test-000431",  # would like to exchange currency
        "banking77-test-000401",  # how can I exchange between currencies
    ],
))

# 70. Currency exchange — capability / supported currencies
clusters.append((
    "What currencies can be exchanged / is exchange supported",
    "User asks whether the app can exchange currency at all or which currencies are supported.",
    [
        "banking77-test-000404",  # will this app exchange currencies
        "banking77-test-000400",  # exchange currencies with this
        "banking77-test-000407",  # exchange currencies with this app
        "banking77-test-000410",  # currencies that can be exchanged
        "banking77-test-000428",  # currencies I can exchange for
        "banking77-test-000429",  # currencies app will exchange
        "banking77-test-000269",  # exchange and hold fiat currencies
        "banking77-test-000271",  # can I exchange currencies
        "banking77-test-000254",  # additional currency options
        "banking77-test-000257",  # supported fiat currencies
        "banking77-test-000261",  # do exchanges of EUR
        "banking77-test-002168",  # buy crypto, app doesn't allow
    ],
))

# 71. Currency exchange — fee / cost
clusters.append((
    "Currency exchange fees / cost",
    "User asks about the cost or fee of doing a currency exchange.",
    [
        "banking77-test-002780",  # extra fee to exchange currency
        "banking77-test-002783",  # exchange fee
        "banking77-test-002799",  # cost to exchange currencies
        "banking77-test-002782",  # charged for exchanging currencies
        "banking77-test-002785",  # how much you charge for exchange
        "banking77-test-002787",  # cost to switch when travel
        "banking77-test-002760",  # exchange fee
        "banking77-test-002769",  # exchange charge
        "banking77-test-002776",  # cross currency exchange cost and discounts
        "banking77-test-002773",  # cost extra to exchange currencies
        "banking77-test-002790",  # extra charges for exchanging currency
        "banking77-test-002793",  # discount for frequent exchanging
    ],
))

# 72. Currency exchange rate — how derived / info
clusters.append((
    "How exchange rates are calculated / sourced",
    "User asks how the exchange rates are set, where they come from, or if it's a good time.",
    [
        "banking77-test-000088",  # how are exchange rates calculated
        "banking77-test-000082",  # how did you come up with exchange rates
        "banking77-test-000083",  # where do you acquire exchange rates
        "banking77-test-000089",  # what are exchange rates based on
        "banking77-test-000113",  # where are you getting exchange rates
        "banking77-test-000109",  # what is the exchange rate
        "banking77-test-000093",  # good time to exchange
    ],
))

# 73. Wrong exchange rate on a card purchase
clusters.append((
    "Wrong exchange rate on a card purchase",
    "User had the wrong exchange rate applied on a card purchase in a foreign currency.",
    [
        "banking77-test-000135",  # bought overseas, wrong exchange rate on statement
        "banking77-test-000158",  # wrong rate on item bought in foreign currency
        "banking77-test-000159",  # wrong exchange rate on purchase in foreign country
        "banking77-test-000146",  # wrong exchange rate after purchasing
        "banking77-test-000149",  # incorrect exchange rate for purchase
        "banking77-test-000154",  # wrong exchange rate on recent purchase
        "banking77-test-000147",  # exchange rate doesn't look right
        "banking77-test-000145",  # exchange rate seems off
        "banking77-test-000144",  # overcharged, exchange rate wrong
        "banking77-test-000137",  # exchange rate for card payment wrong
        "banking77-test-002581",  # too much taken during currency exchange
        "banking77-test-000132",  # rate for currency exchange wrong when bought
        "banking77-test-000129",  # swapped Russian Ruble for GBP, charged too much
    ],
))

# 74. Receive salary in different currency / payroll
clusters.append((
    "Receive salary / paycheck in a particular currency",
    "User asks about receiving salary, paycheck, or wages, often in a non-default currency.",
    [
        "banking77-test-002263",  # received salary in GBP, change to my currency
        "banking77-test-002247",  # paid in GBP, configure where
        "banking77-test-002241",  # how to get paid in different currency
        "banking77-test-002242",  # receive salary in different currency
        "banking77-test-002246",  # choose GBP for salary deposit
        "banking77-test-002249",  # use this to receive salary
        "banking77-test-002279",  # paycheck through here
    ],
))

# 75. Transfer not arrived / pending
clusters.append((
    "Money transfer not arrived / pending",
    "User initiated a transfer that hasn't arrived or is still pending; status check rather than declined.",
    [
        "banking77-test-002292",  # transferred funds, didn't go through
        "banking77-test-002704",  # transfer not showing up
        "banking77-test-002689",  # bank transfer balance didn't update
        "banking77-test-001855",  # transferred yesterday, not available
        "banking77-test-001877",  # transfer hasn't gone through
        "banking77-test-001878",  # transfer still pending
        "banking77-test-001879",  # how long for money transfer to show
        "banking77-test-001849",  # transfer pending, why so long
        "banking77-test-001845",  # transferred but still pending
        "banking77-test-002693",  # made transfer, still waiting
        "banking77-test-002684",  # when will balance update after transfer
        "banking77-test-002296",  # transfer will not go through
        "banking77-test-002712",  # bank transfer hasn't show up
        "banking77-test-002713",  # where is my transfer from country
        "banking77-test-000848",  # transfer hasn't arrived
        "banking77-test-000851",  # where is the transfer I started
        "banking77-test-000842",  # waiting for transaction to complete
        "banking77-test-000844",  # sent money but recipient hasn't received
        "banking77-test-000862",  # transfer not received by friend
        "banking77-test-000874",  # how long for transfers, friend hasn't gotten
        "banking77-test-002214",  # recipient got less, need to transfer difference (problem with sent transfer)
    ],
))

# 76. Transfer declined / failed
clusters.append((
    "Money transfer declined or failed",
    "User's money transfer was declined, blocked, or repeatedly failed.",
    [
        "banking77-test-001722",  # continuous failure for all transfers
        "banking77-test-002181",  # couldn't do a transfer
        "banking77-test-002164",  # had a transfer blocked
        "banking77-test-001745",  # transfer was declined
        "banking77-test-001746",  # transfer was declined
        "banking77-test-001747",  # why was transfer declined
        "banking77-test-001743",  # reason transfer was declined
        "banking77-test-001732",  # why did I get transfer declined
        "banking77-test-001733",  # why did transfer get declined
        "banking77-test-002188",  # beneficiary denied
        "banking77-test-002170",  # can't transfer to beneficiary
        "banking77-test-002193",  # not able to do transfer to beneficiary
        "banking77-test-002160",  # tried numerous times, why not going through
        "banking77-test-002287",  # transfer keeps getting returned
        "banking77-test-002297",  # transfer keeps failing
        "banking77-test-002299",  # why would transfer fail
        "banking77-test-002309",  # continuously facing failed transfers
        "banking77-test-002312",  # standard transfer 5 times, system broken
    ],
))

# 77. (merged) — "recipient got lower amount" folded into transfer-pending (banking77-test-002214 moved there)


# 78. Transfer — how to / process / using credit card
clusters.append((
    "How to make a money transfer (incl. via credit card)",
    "User asks how to initiate a money transfer, doesn't understand the process, or wants to use a credit card for transfer.",
    [
        "banking77-test-001352",  # how do I transfer using credit card
        "banking77-test-001328",  # use credit card to transfer money
        "banking77-test-001335",  # transfer money using credit card
        "banking77-test-001345",  # use credit card to transfer
        "banking77-test-002324",  # don't understand money transfer process
        "banking77-test-000601",  # would like to make a transfer, cost
    ],
))

# 79. Transfer — supported types (SWIFT/SEPA) / policy
clusters.append((
    "Supported transfer types and policy (SWIFT/SEPA)",
    "User asks about supported transfer types like SWIFT or SEPA, or the transfer policy generally.",
    [
        "banking77-test-000612",  # SWIFT transfers
        "banking77-test-000609",  # transfer from SWIFT
        "banking77-test-000632",  # transfer policy
        "banking77-test-002257",  # different ways for someone to send me money
    ],
))

# 80. Transfer — fees / cost
clusters.append((
    "Transfer fees and costs",
    "User asks about being charged fees on transfers, including receiving and SEPA fees.",
    [
        "banking77-test-002224",  # why am I seeing a transfer fee
        "banking77-test-002229",  # why was my transfer charged fees
        "banking77-test-002238",  # charged extra fee when transferring
        "banking77-test-002203",  # charged just for transferring
        "banking77-test-002223",  # sent money but charged extra
        "banking77-test-000628",  # charged for SEPA transfer
        "banking77-test-000637",  # charged if getting money
        "banking77-test-000638",  # total cost of transfer
        "banking77-test-000620",  # charges for receiving money
        "banking77-test-001875",  # transfer pending today, charged a fee?
    ],
))

# 81. Transfer — time to process / by country
clusters.append((
    "Transfer processing time (by country / region)",
    "User asks how long a transfer will take, often from a specific country/region.",
    [
        "banking77-test-002067",  # bank transfer from Europe, how long
        "banking77-test-002068",  # how long process transfers from Europe
        "banking77-test-002076",  # transfer from Europe how long
        "banking77-test-002041",  # wait time for transfer from US
        "banking77-test-002042",  # fast transfer from China, how fast
        "banking77-test-002057",  # how long for funds from US
        "banking77-test-002062",  # how many days until money in account
    ],
))

# 82. Cancel a transfer / transaction
clusters.append((
    "Cancel a transfer or transaction",
    "User wants to cancel a transfer or a pending transaction.",
    [
        "banking77-test-000698",  # cancel a transfer
        "banking77-test-000693",  # cancel my transaction
        "banking77-test-000716",  # cancel a transaction
        "banking77-test-000682",  # cancel my transaction
    ],
))

# 83. Direct debit — not mine / disputed
clusters.append((
    "Direct debit not recognized / not mine",
    "User sees a direct debit on their account they didn't authorize or recognize.",
    [
        "banking77-test-001502",  # Direct Debit payment may not be right
        "banking77-test-001508",  # didn't make this direct debit transaction
        "banking77-test-001481",  # direct debit that's not mine
    ],
))

# 84. Refund — request / how to
clusters.append((
    "How do I get a refund / request a refund",
    "User wants to know how to get a refund, or is requesting one for a purchase.",
    [
        "banking77-test-000165",  # refund on extra pound charged
        "banking77-test-001697",  # need a refund
        "banking77-test-001717",  # process refund for purchase
        "banking77-test-001700",  # how do I get refunded
        "banking77-test-001684",  # need to do a refund
        "banking77-test-001713",  # bought item, wrong amount, can I get refund
        "banking77-test-001698",  # unhappy with purchase, how to cancel
        "banking77-test-001710",  # cancel order and process refund
        "banking77-test-001711",  # reverse a purchase, cancel
    ],
))

# 85. Refund — status check / missing
clusters.append((
    "Where is my refund — status check",
    "User has already initiated/expected a refund and wants to know where it is.",
    [
        "banking77-test-001778",  # where can I see the refund
        "banking77-test-001786",  # missing a refund
        "banking77-test-001766",  # awaiting my refund
        "banking77-test-001783",  # statement doesn't show refund processed
    ],
))

# 86. Double / duplicate charge
clusters.append((
    "Charged twice / duplicate charge on a transaction",
    "User was charged multiple times for the same transaction and wants it investigated/refunded.",
    [
        "banking77-test-001990",  # charged multiple times for same transaction
        "banking77-test-001992",  # multiple charges on same transaction
        "banking77-test-001962",  # charged twice for a restaurant
        "banking77-test-001973",  # double charged this week
        "banking77-test-001970",  # check if charged twice
        "banking77-test-001971",  # charged twice
        "banking77-test-001975",  # charge showed up twice
        "banking77-test-001995",  # charged twice from restaurant
        "banking77-test-001998",  # payment charged twice instead of once
    ],
))

# 87. Unrecognized small / extra charge / "pending" fee
clusters.append((
    "Unrecognized small extra charge on statement",
    "User questions a specific small/extra charge (often $1/pound) on their statement.",
    [
        "banking77-test-000161",  # extra pound fee on statement
        "banking77-test-000173",  # where did this fee come from
        "banking77-test-000185",  # extra fee on statement where from
        "banking77-test-000187",  # extra fee on statement
        "banking77-test-000168",  # statement has extra charges
        "banking77-test-000166",  # explain random $1 charge
        "banking77-test-000191",  # transaction for $1, why
        "banking77-test-000172",  # when will $1 transaction credit
        "banking77-test-000197",  # what is the 1 euro fee for
        "banking77-test-000192",  # weird pound charge, pending two days
        "banking77-test-001104",  # payment listed in error
        "banking77-test-002213",  # fees increasing while abroad
        "banking77-test-001504",  # dispute payment several weeks after (older unrecognized charge)
    ],
))

# 88. (merged) — "dispute older charge" folded into unrecognized small/extra charge


# 89. Cash / cheque deposit not appearing
clusters.append((
    "Cash or cheque deposit not showing up",
    "User made a deposit (cash or cheque) and it has not appeared in their balance.",
    [
        "banking77-test-001060",  # where is my deposit
        "banking77-test-001066",  # balance didn't increase after depositing check
        "banking77-test-001065",  # deposit of cash and check not added
        "banking77-test-001064",  # cash deposit not showing
        "banking77-test-001061",  # statement doesn't show cash deposit
        "banking77-test-001050",  # cash deposit not appeared
        "banking77-test-001063",  # balance didn't update after cash/cheque
        "banking77-test-001068",  # cash gone after tried to deposit
        "banking77-test-001052",  # deposit this morning, still pending
        "banking77-test-001042",  # account not updating after check deposit
        "banking77-test-001073",  # has check deposited cleared
    ],
))

# 90. Incoming payment (deposit) — when will it arrive
clusters.append((
    "Incoming payment processing on a deactivated/special account",
    "User asks about whether an incoming payment will be processed (e.g. into a deactivated account).",
    [
        "banking77-test-000246",  # incoming payment, account deactivated, still processed
    ],
))

# 91. Card visa/mastercard brand preference
clusters.append((
    "Card brand — Visa vs Mastercard preference / availability",
    "User asks whether they can choose between Visa and Mastercard, or expresses a brand preference.",
    [
        "banking77-test-001295",  # choose between Visa and Mastercard
        "banking77-test-001310",  # like Mastercard better
        "banking77-test-001298",  # want Visa and Mastercard
        "banking77-test-001289",  # can I get a mastercard
        "banking77-test-001303",  # apply for visa card
        "banking77-test-001318",  # do you use Mastercard or Visa
    ],
))

# 92. Supported cards / currencies for top-up source
clusters.append((
    "Supported card brands / currencies for adding money",
    "User asks what cards and currencies are accepted for adding money / topping up.",
    [
        "banking77-test-000903",  # what currencies accepted to add money
        "banking77-test-000913",  # any currency to add money
        "banking77-test-000896",  # what credit cards supported
        "banking77-test-000904",  # why no support for AmEx
        "banking77-test-000884",  # currencies for adding money
        "banking77-test-000889",  # what cards and currencies
        "banking77-test-000901",  # forms of currency accepted
        "banking77-test-000905",  # which cards and what currency supported
    ],
))

# 93. Where can my card be used (merchants/online)
clusters.append((
    "Where / by whom my card is accepted",
    "User asks where they can use their card — merchants, online, regions, acceptance.",
    [
        "banking77-test-000999",  # who accepts my card
        "banking77-test-000977",  # where can I use my card
        "banking77-test-000964",  # where can the card be used
        "banking77-test-000978",  # which outlets accept my card
        "banking77-test-000980",  # what stores will take card
        "banking77-test-000981",  # any place I cannot use card
        "banking77-test-000983",  # card usable anywhere
        "banking77-test-000991",  # online purchases with card
        "banking77-test-000995",  # any business take this card
        "banking77-test-000908",  # options for payment
    ],
))

# 94. (merged) — generic "reinstall payment app" folded into card-not-working/diagnose generic


# ---------------------------------------------------------------------------

# Sanity check coverage
covered: dict[str, str] = {}
duplicates: list[tuple[str, str, str]] = []
for name, _desc, ids in clusters:
    for i in ids:
        if i in covered:
            duplicates.append((i, covered[i], name))
        else:
            covered[i] = name

missing = sorted(all_ids - set(covered))
extras = sorted(set(covered) - all_ids)

if duplicates:
    print(f"[ERROR] {len(duplicates)} duplicates:")
    for i, a, b in duplicates[:20]:
        print(f"  {i}: {a!r} vs {b!r}")
    sys.exit(1)
if extras:
    print(f"[ERROR] {len(extras)} ids in clusters not in sample (first 20):")
    for i in extras[:20]:
        print(f"  {i}")
    sys.exit(1)

print(f"clusters: {len(clusters)}")
print(f"covered: {len(covered)} / {len(all_ids)}")
print(f"missing: {len(missing)}")
if missing:
    print("First 30 missing:")
    for i in missing[:30]:
        print(f"  {i}: {id_to_text[i][:80]!r}")
    sys.exit(2)

# Build proposal payload
ts_now = dt.datetime.now()
timestamp_str = ts_now.strftime("%Y-%m-%d %H:%M:%S")
file_ts = ts_now.strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:8]
out_path = OUT_DIR / f"prop_{file_ts}_{uid}.json"

payload = {
    "timestamp": timestamp_str,
    "sample_size": 600,
    "sample_strategy": "shared",
    "style": "fine-grained",
    "existing_clusters_considered": False,
    "clusters": [
        {
            "name": name,
            "description": desc,
            "text_ids": ids,
            "reasoning": f"Grouped {len(ids)} texts whose specific intent matches {name!r}.",
        }
        for name, desc, ids in clusters
    ],
    "unclustered_ids": [],
    "observations": (
        "Targeted ~85 fine-grained customer intents (final: {n}). Aggressive splits applied along the requested axes: "
        "status-check vs action-request (e.g. 'where is my refund' vs 'I need a refund', 'top-up pending' vs 'top-up failed'), "
        "failed-attempt vs information question (e.g. 'transfer declined' vs 'how to make a transfer', "
        "'card payment declined' vs 'where can I use my card'), and lost/stolen separated from generic phrasing. "
        "Top-up split by funding source (cheque, cash, bank-transfer, Apple Pay, Google Pay/wearable, card, AmEx, AmEx+ApplePay-failure). "
        "Card lifecycle split into: not-arrived, delivery-time, delivery-location, tracking, ordering, fees, multiples, activation, expiring, reactivation. "
        "ATM split into: declined, no-cash-dispensed, wrong-amount, swallowed-card, fee-on-withdrawal, wrong-exchange-rate-on-withdrawal, locator, "
        "which-ATMs-accept, post-time, unauthorized-withdrawal. Every text id assigned (unclustered_ids=[]); zero leftover bucket. "
        "Borderline calls: minor PIN texts ('change PIN' vs 'reset PIN' kept together as one 'change/set' intent since intent is identical; "
        "'don't have / can't find' separated from 'change'). Single-text 'recipient got less' and 'reinstall app' kept as standalone fine-grained intents."
    ).format(n=len(clusters)),
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print(f"wrote: {out_path}")
print(f"cluster_count: {len(clusters)}")
