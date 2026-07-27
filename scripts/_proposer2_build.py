"""Build proposer #2 proposal -- ~72 clusters, balanced grouping/splitting."""
import json, uuid, datetime, pathlib

SAMPLE = pathlib.Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\banking77\seed=0_auditdiv\samples\shared_proposer_sample.json")
OUT_DIR = pathlib.Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\banking77\seed=0_auditdiv\proposals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with SAMPLE.open() as f:
    data = json.load(f)

by_id = {d["id"]: d["text"] for d in data}
assert len(by_id) == 600

clusters = []

def C(name, description, ids, reasoning):
    seen = set()
    uniq = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    clusters.append({
        "name": name,
        "description": description,
        "text_ids": uniq,
        "reasoning": reasoning,
    })

def s(*shorts):
    return [f"banking77-test-{x}" for x in shorts]

# ===== CARDS =====
C("card_arrival_status",
  "Customer is asking about the status, tracking, or expected arrival of a card that has been ordered or shipped but not yet received, including app-visibility of a freshly received card.",
  s("000004","000312","000013","000036","000032","000009","000006","000039","000057","000066","000282"),
  "Waiting for a card / tracking, plus 'I received it but the app doesn't show it' and 'I need my card quick' (all card-arrival-impatience).")

C("card_delivery_time_and_destination",
  "Questions about how long card delivery takes or where the card can be shipped to.",
  s("000302","000286","000290","000303","000294","002506","002509","002491"),
  "Delivery-time and delivery-location questions, distinct from order status.")

C("order_physical_card",
  "Request to obtain a physical (non-virtual) card via the app, including general 'how do I get one' questions and fees/cost for getting a physical card.",
  s("002502","002480","002483","002488","002497","002499","002505","002507"),
  "Explicit requests for a physical card, plus the physical-card-fee questions (banking77 'getting_a_physical_card' covers both).")

C("get_physical_card_abroad",
  "Asking whether or how to receive a card while living outside the issuer's home country.",
  s("003059","003045","003048","003061","003054"),
  "Cross-border card-issuance questions; its own banking77 intent.")

C("get_disposable_virtual_card",
  "Asking how to obtain or set up a disposable virtual card.",
  s("002623","002626","000921","000956"),
  "Specifically about acquiring a disposable or virtual card.")

C("explain_disposable_virtual_card",
  "Asking for an explanation of what disposable or virtual cards are and how they work.",
  s("002600","002610","002616","002602","002638","002622"),
  "Explanatory or definition queries about disposable virtual cards.")

C("disposable_card_limits",
  "Questions about quantity, usage, daily, or transaction limits for disposable virtual cards.",
  s("001362","001364","001373","001365","001369","001372","001375","001380","001382","001383","001385","001393","001368"),
  "All limit-related questions for disposable cards including 'how many can I have'.")

C("multiple_physical_cards",
  "Asking whether the account can have multiple physical cards (for example for family members) or how many cards per account, including extra-card fees.",
  s("002406","002417","002401","002435","002404"),
  "Multi-card and extra-card requests.")

C("activate_or_reactivate_card",
  "Customer asking how to activate a newly received card, requesting activation, or reactivating a card that was previously found after being lost or frozen.",
  s("002864","002857","002844","002868","002871","002872","002879","000076","000062"),
  "Activation and reactivation collapsed — both are 'turn this card on'.")

C("card_acceptance_locations",
  "Asking generally where the card can be used or which merchants or businesses accept it, including online use.",
  s("000977","000999","000980","000964","000981","000983","000995","000991","000978"),
  "Where-can-I-use-this-card questions covering both in-store and online scope.")

C("link_external_card",
  "Asking how to link a (typically credit) card to the app or account.",
  s("000054","000050","000060","000058"),
  "Linking an external card to the account.")

C("supported_card_types",
  "Questions about which card brands (Visa, Mastercard, Amex) are supported, including choosing between them.",
  s("001295","001310","001298","001289","001303","001318","000896","000904"),
  "Card-type and brand support questions; distinct from currency support.")

C("card_replacement_or_renewal",
  "Asking about getting a new card to replace an expiring, broken, or otherwise to-be-renewed card, including cost and timing.",
  s("002947","002926","002925","002923","002943","002927","002931","002946","002959","002415"),
  "Lifecycle replacement covering expiry, ordering before expiry, and renewal cost/timing.")

# ===== CARD FAILURES (non-ATM) =====
C("card_payment_declined",
  "Customer's card payment was declined, generally or for a specific transaction, and they want to understand why.",
  s("001822","000375","000373","001837","001813","001824","001834","001828","001811","001835","002083","002080","001502","000367","000371","000388","002544"),
  "Card-payment declined, not working, or cancelled queries.")

C("contactless_not_working",
  "Customer's contactless (NFC) card or phone payment isn't working.",
  s("000570","000572","000583","000567","000577"),
  "Contactless-failure intent.")

C("card_broken_or_damaged",
  "Customer reports a physical card that is broken or damaged.",
  s("000382","000374","000383"),
  "Physically broken card distinct from declined.")

C("virtual_card_not_working",
  "Virtual or disposable card was declined or failed at a merchant.",
  s("002532","002541","002543","002553"),
  "Virtual-card-specific failure, separate from general card decline.")

# ===== ATM =====
C("atm_withdrawal_declined",
  "Withdrawal at an ATM was declined or refused entirely.",
  s("001577","001576","001565","001570","001571","001573","001583","001589","001591"),
  "ATM-decline specifically, not partial-amount issues.")

C("atm_wrong_amount_dispensed",
  "ATM dispensed less cash than requested, or app shows the wrong amount versus what was received.",
  s("000782","000775","000764","000779","000787","000797","000770","000772","000786","000780","000798"),
  "Wrong or partial cash dispensed; banking77 wrong_amount_of_cash_received intent.")

C("atm_swallowed_card",
  "ATM trapped or kept the customer's card.",
  s("001952","001933","001951","001953","001926","001928","001931","001941","001948","001958"),
  "Card-stuck-in-ATM intent.")

C("find_nearby_atm",
  "Asking for nearby ATMs or where one can withdraw money / use the card at an ATM.",
  s("001442","001462","001463","001466","001468","001471","001478","001449","001444"),
  "ATM-locator and which-ATMs-accept-this-card intent.")

C("atm_withdrawal_fee_charged",
  "Customer was charged a fee for an ATM cash withdrawal, is questioning it, or is asking up-front whether there will be a fee/extra cost on a withdrawal.",
  s("002909","002889","002882","002910","002893","002895","002894","002892","002898","002900","002907","002916","002880","002888","002575","002597"),
  "Cash-withdrawal-fee complaints and the up-front 'will I be charged' question.")

C("cash_withdrawal_wrong_exchange_rate",
  "Customer used an ATM in a foreign currency and was charged the wrong exchange rate.",
  s("002561","002595","002593","002567","002570"),
  "FX-rate-on-ATM-withdrawal complaint, distinct from general FX-rate issue.")

# ===== PIN =====
C("pin_lookup_or_retrieval",
  "Customer can't find or has forgotten their PIN and wants to retrieve it.",
  s("001270","001242","001278","001261","000521","001240","001275"),
  "Locating or retrieving an existing PIN.")

C("pin_change_or_reset",
  "Customer wants to change, update, or reset their PIN.",
  s("002135","002148","002137","000552","002133","002125","002150","002153","002134","002142","002121"),
  "Changing/setting/resetting PIN.")

C("pin_blocked_too_many_attempts",
  "Customer's PIN is blocked because of too many wrong attempts and they want to unblock it.",
  s("000533","000546","000531","000534","000520"),
  "PIN blocked or locked.")

# ===== PASSWORD / APP ACCESS =====
C("password_or_passcode_reset",
  "Customer forgot or can't enter their app passcode or password and wants to reset it.",
  s("001550","001523","001525","001538","001520","001552","001548"),
  "App passcode and password reset.")

C("account_locked_blocked",
  "Account is blocked and the customer can't log in or wants to know why.",
  s("000542","000543"),
  "Account-login block, distinct from PIN block.")

# ===== IDENTITY VERIFICATION =====
C("identity_verification_problem",
  "Customer is having trouble passing identity verification or the app doesn't recognize them.",
  s("001234","001238","001239","001233","001236","001213"),
  "Failing to verify ID.")

C("identity_verification_methods",
  "Asking what methods or documents are available for verifying identity, where to verify, and how long the verification takes.",
  s("001192","003017","003020","003003","003025","003036","001225","001230"),
  "Methods, process, location, and timing of identity verification.")

C("identity_verification_required",
  "Asking whether identity verification is mandatory, expressing refusal, or asking what one can do without it.",
  s("001178","001189","001186","001187","001175","001185","001179","001201","001205"),
  "Conscientious objection or whether verification is mandatory.")

# ===== ACCOUNT MANAGEMENT =====
C("update_personal_details",
  "Customer wants to change personal information such as address, name, or contact details.",
  s("001154","001128","001127","001150","001140","001146","001125"),
  "Change-name / change-address / change-personal-details intent.")

C("close_or_delete_account",
  "Customer wants to close, delete, or cancel their account.",
  s("001913","001912","001887","001909"),
  "Account-closure intent.")

C("age_to_open_account",
  "Asking about the minimum or required age to open an account.",
  s("000511","000509","000507","000487","000480","000490"),
  "Age-limit-for-account intent.")

C("country_availability",
  "Asking which countries the service operates in or whether their country is supported.",
  s("003069","003056","003058"),
  "Country / location availability of the service.")

# ===== LOST/STOLEN / FRAUD / FREEZE =====
C("lost_or_stolen_card",
  "Customer reports a card as lost or stolen and asks for help or replacement.",
  s("000446","000479","000475","000471","000470","000445","000474","000454","000472"),
  "Lost or stolen card intent including reporting and replacement.")

C("phone_lost_lock_account",
  "Customer lost their phone (or it was stolen / they were mugged) and wants to secure the account.",
  s("001658","001647","001645","001662","001651","001671","001678"),
  "Lost-phone account-security intent.")

C("freeze_card",
  "Customer asks to freeze their card, often via the app or as a precaution.",
  s("001421","001407"),
  "Freeze-card intent specifically.")

C("unrecognized_card_transaction_fraud",
  "Customer sees a card payment or charge on their account they did not make and suspects fraud.",
  s("001110","001111","001402","001434","001516","001419"),
  "Card-payment-not-recognised (fraud) intent.")

C("unrecognized_cash_withdrawal_fraud",
  "Customer sees a cash withdrawal in the app or account they didn't make or authorize.",
  s("002736","002726","002758","002759","002744","002756","002725","002737","002739","002732","002724","002755"),
  "Cash-withdrawal-not-recognised intent.")

C("unrecognized_direct_debit",
  "Customer sees a direct debit they don't recognize.",
  s("001481","001508"),
  "Direct-debit-not-recognised intent.")

# ===== TOP-UP METHODS / LIMITS / FEES / SETUP =====
C("topup_methods_overview",
  "Asking generally how to top up, which methods are available, who is allowed to top up, or unusual top-up source questions.",
  s("001332","002466","000881","001354","001326"),
  "General topup how-to / methods-overview, including 'who can top up' and odd phrasings.")

C("topup_by_card",
  "Asking whether or how to top up using a debit, credit, or bank card.",
  s("001320","002829"),
  "Topup-by-card method.")

C("topup_by_bank_transfer",
  "Asking about topping up via bank transfer.",
  s("002359","002335","002350"),
  "Topup-by-bank-transfer specifically.")

C("topup_by_cheque",
  "Asking about topping up by cheque.",
  s("002442","002465","002473"),
  "Topup-by-cheque method.")

C("topup_by_cash",
  "Asking how or where to top up using cash, including pay-by-check overlap.",
  s("002470","002443","002444","002458","002450","002468","002451"),
  "Topup-by-cash method.")

C("topup_by_apple_or_google_pay",
  "Asking about topping up via Apple Pay, Google Pay, or similar mobile wallet.",
  s("002980","002989","002967","002969","002990","002999"),
  "Topup via mobile-wallet methods.")

C("topup_limit",
  "Asking about minimum or maximum amount limits on top-ups.",
  s("000753","000730","000735","000738","000745","000746","000757"),
  "Topup-limits questions.")

C("topup_fee",
  "Asking about fees for topping up by transfer, card, or other methods.",
  s("000618","002825","000619","000607","002816","002817","002801"),
  "Topup-fee questions across methods.")

C("auto_topup",
  "Asking how to set up or find the automatic top-up feature, including scheduled or interval top-ups, and about restrictions/limits on auto top-up.",
  s("000323","000329","000328","000330","000333","000342","000341","000346","000356","000334","000337"),
  "Auto-topup setup, scheduling, and limits combined (single banking77 'automatic_top_up' intent in practice).")

# ===== TOP-UP PROBLEMS =====
C("topup_failed",
  "Customer reports their top-up failed (including via Apple Pay, Google Pay, or American Express) and wants help understanding why.",
  s("001026","001027","002649","002652","002659","002665","002677","002966","002641","002668","002658","000894","002991","002978","002972"),
  "Top-up failed across methods, including Amex/Apple-Pay-Amex failures which surface as the same intent.")

C("topup_pending",
  "Customer's top-up is stuck in 'pending' status and they want it resolved.",
  s("000645","000648","000655","000663","000671","000678","000658","000670"),
  "Pending top-up status complaint.")

C("topup_verification_code",
  "Customer needs the verification code for their top-up card, or asks why verification is required.",
  s("002389","002395","002397","002387","002374"),
  "Top-up card verification code.")

# ===== TRANSFERS =====
C("how_to_transfer",
  "General questions about how to make a transfer, understanding the transfer process, using a credit card to transfer, or asking about available payment options.",
  s("002324","002357","002333","002331","001335","001352","001328","001345","002327","000908"),
  "How-to-transfer plus broad payment-options questions.")

C("supported_transfer_types",
  "Asking about supported transfer protocols (SWIFT, SEPA) and the transfer policy, including fee for protocol-specific transfers.",
  s("000612","000609","000632","000628"),
  "Supported-transfer-network questions including SEPA-fee specifically.")

C("transfer_not_received_or_pending",
  "Customer's transfer (sent or received) hasn't shown up, the balance hasn't updated, or it has been stuck in pending status for too long; includes questions about whether incoming payments will process at all.",
  s("002292","002689","002704","001855","000844","000851","000862","000874","000842","000848","001877","002693","002713","002712","002684","000246","001845","001878","001849"),
  "Transfer not arriving / pending / balance not updating; same underlying customer state regardless of pending-vs-stuck framing.")

C("transfer_or_withdrawal_timing",
  "Asking how long a transfer takes (especially by origin country or region) or how long an ATM withdrawal takes to post.",
  s("002041","002042","002057","002062","002068","002076","002067","001879","000204","000237"),
  "Money-movement timing questions: transfers from various origins and ATM-withdrawal-posting time.")

C("transfer_failed_or_declined",
  "Customer's transfer outright failed or was declined, including beneficiary-rejection issues.",
  s("001722","002181","002296","002297","002299","002309","002160","002164","001745","001746","001733","001743","001747","001732","002287","002312","002170","002193","002188"),
  "Failed and declined transfer including beneficiary issues.")

C("transfer_fees_charged",
  "Customer was charged a fee for a transfer and is asking why or what the cost was.",
  s("002224","002229","002238","002203","002223","000638","000601","001875","002214"),
  "Transfer-fee complaints and transfer-cost questions.")

# ===== RECEIVING / INCOMING =====
C("receive_money_methods_and_fees",
  "Asking how others can send money to this account and whether there are fees for receiving money.",
  s("002257","000637","000620"),
  "Receive-money topic combining methods and fees.")

C("salary_via_account",
  "Asking whether or how the account can receive salary, including currency configuration for salary.",
  s("002263","002247","002242","002241","002246","002249","002279"),
  "Salary-into-account intent including currency configuration.")

# ===== EXCHANGE =====
C("currency_exchange_how_to",
  "Asking how to exchange currencies in the app, or general 'can I exchange?' / 'I'd like to exchange' requests.",
  s("000404","000412","000400","000401","000407","000413","000415","000431","000255","000250","000268","000261","000271","002168"),
  "How-to or can-I exchange currencies, including specific to-currency requests.")

C("supported_exchange_currencies",
  "Asking which currencies are supported for exchange or holding, including for top-up.",
  s("000257","000254","000269","000410","000428","000429","000901","000884","000903","000913","000905","000889"),
  "Supported-currencies question.")

C("exchange_fee",
  "Asking whether there is a fee for exchanging currencies and how much.",
  s("002780","002783","002782","002785","002790","002787","002793","002773","002760","002776","002799","002769"),
  "FX-charge / exchange-fee questions.")

C("exchange_rate_source",
  "Asking how exchange rates are calculated, where they come from, or whether it's a good time to exchange.",
  s("000088","000089","000082","000083","000093","000109","000113"),
  "Exchange-rate-source / methodology questions.")

C("wrong_exchange_rate_on_purchase",
  "Customer was given the wrong exchange rate on a purchase or card payment, or feels too much money was taken during a currency exchange.",
  s("000135","000158","000149","000132","000137","000144","000146","000147","000154","000159","000145","000129","002581"),
  "Wrong-exchange-rate-for-payment complaint, including overcharged-on-FX-swap (same underlying complaint).")

# ===== FEES (CARD PAYMENT / UNKNOWN) =====
C("fee_card_payment",
  "Customer was charged a fee specifically for a card payment and wants to know why or how much.",
  s("000813","000816","000826","000827","000810","000832","000811","000833","000837"),
  "Card-payment-fee questions.")

C("extra_unknown_fee_on_statement",
  "Customer sees an unexpected, unknown fee or extra charge on their statement.",
  s("000173","000185","000187","002213","000168","000161","000166","000197"),
  "Unexplained extra fee on statement.")

# ===== REFUNDS / DISPUTES / CANCEL =====
C("request_refund",
  "Customer is asking to get a refund for a purchase or charge.",
  s("000165","001697","001700","001717","001713","001684"),
  "Refund-request intent.")

C("refund_status_or_missing",
  "Customer asks about the status of a pending refund or notes that an expected refund hasn't shown up.",
  s("001778","001786","001766","001783"),
  "Refund-not-showing / refund-status.")

C("cancel_transaction_or_transfer",
  "Customer wants to cancel a transaction, transfer, or order they made, or notes that a card payment was cancelled.",
  s("000693","000716","000682","001710","001711","001698","002115","000698"),
  "Cancel intent across transactions and transfers.")

C("dispute_double_charge",
  "Customer was charged twice (or multiple times) for the same transaction and wants it corrected.",
  s("001990","001992","001962","001970","001971","001973","001995","001998","001975"),
  "Double-charge dispute.")

C("dispute_random_dollar_charge",
  "Customer sees a small ($1 / EUR1) unexplained pending charge and wants it explained, including dispute-after-the-fact.",
  s("000191","000192","000172","001504","001104"),
  "Small random charge (often verification pre-auth) and 'payment listed in error' / late dispute.")

C("pending_card_payment",
  "Customer's card payment is stuck in pending status for too long, or they can't access funds because a transaction is still pending.",
  s("001602","001624","001630","001613","001617","001633","000227"),
  "Pending-card-payment intent including 'can't access funds due to pending'.")

# ===== DEPOSITS =====
C("cash_or_check_deposit_missing",
  "Customer made a cash or check deposit and it's not showing in their balance, including whether it has cleared.",
  s("001060","001066","001065","001068","001064","001063","001061","001050","001073","001042","001052"),
  "Pending-cash-deposit / balance-not-updated-after-deposit.")

# ===== SOURCE OF FUNDS =====
C("source_of_funds_lookup_or_requirement",
  "Customer wants to view source-of-funds info, or asks why the bank needs source-of-funds details.",
  s("002021","002017","002036","002001","002025","002004","002038","002008","002014","002029"),
  "Verify-source-of-funds intent both lookup and the rationale question.")

# ===== EDGE / MISC =====
C("payment_app_general_troubleshoot",
  "General app-level malfunctions like 'app refused payment', asking whether to reinstall the app, or where to locate the card within the app.",
  s("002094","000599","000000"),
  "Catch-all app-level glitches not tied to a specific channel; small but conceptually distinct from card_payment_declined.")


# Write proposal file
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:8]
out_path = OUT_DIR / f"prop_{ts}_{uid}.json"

proposal = {
    "timestamp": datetime.datetime.now().isoformat(),
    "sample_size": 600,
    "sample_strategy": "shared",
    "style": "balanced",
    "existing_clusters_considered": False,
    "clusters": clusters,
    "unclustered_ids": [],
    "observations": "Targeted ~72 clusters; landed at the count printed below. Balanced approach: distinct customer actions kept as distinct intents, but very thin singletons folded into the closest semantic neighbour where the underlying action is the same (e.g. 'reactivate found card' folded into 'activate card', 'cancel transfer' into 'cancel transaction', 'transfer pending too long' into 'transfer not received'). Kept finer splits where banking77 has historical evidence of distinct intents (e.g. ATM fee vs. card-payment fee; lost/stolen vs. freeze vs. lost-phone). All 600 sample texts assigned to a real intent cluster with no leftover bucket."
}

with out_path.open("w", encoding="utf-8") as f:
    json.dump(proposal, f, indent=2, ensure_ascii=False)

print(f"Wrote proposal: {out_path}")

# Coverage check
all_ids_in_clusters = []
for c in clusters:
    all_ids_in_clusters.extend(c["text_ids"])

unique_assigned = set(all_ids_in_clusters)
dupes = sorted({x for x in all_ids_in_clusters if all_ids_in_clusters.count(x) > 1})
missing = sorted([k for k in by_id if k not in unique_assigned])
extra = sorted([k for k in unique_assigned if k not in by_id])

print(f"Clusters: {len(clusters)}")
print(f"Assigned (with possible dupes): {len(all_ids_in_clusters)}")
print(f"Unique assigned: {len(unique_assigned)}")
print(f"Duplicates: {len(dupes)}")
for d in dupes:
    print(f"  DUPE: {d}")
print(f"Missing: {len(missing)}")
for m in missing:
    print(f"  MISSING {m}: {by_id[m]!r}")
print(f"Extra: {len(extra)}")
for e in extra:
    print(f"  EXTRA {e}")
