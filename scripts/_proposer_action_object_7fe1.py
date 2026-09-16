"""Build action_object cluster proposal for banking77 (proposer angle: action+object)."""
import json
import os
from datetime import datetime

TS = "20260607_065707"
UUID = "7fe1"
WORKSPACE = os.environ["CLUSTERING_WORKSPACE"]
TEMP = os.path.join(WORKSPACE, "proposals", f"_tmp_{TS}_{UUID}.jsonl")
OUT = os.path.join(WORKSPACE, "proposals", f"prop_{TS}_{UUID}.json")


def b(s):
    return f"banking77-test-{s}"


clusters = [
    # ---- TRANSFER ----
    {
        "name": "explain_declined_transfer",
        "description": "Customer asks why an outgoing money transfer was declined / rejected / failed to send.",
        "text_ids": [b(x) for x in ["001745", "001723", "001722", "001749", "001744", "001733", "002301", "002317", "002305"]],
        "reasoning": "Verb=explain/diagnose, object=outgoing transfer that was declined. Distinct from declined cash withdrawal (different object) and declined card payment (different object).",
    },
    {
        "name": "explain_pending_transfer",
        "description": "Customer asks why a transfer is still pending / hasn't completed yet / shows as pending.",
        "text_ids": [b(x) for x in ["001849", "001858", "001863", "001865", "001856", "002716", "000841"]],
        "reasoning": "Verb=explain status, object=pending transfer. Different from card-payment pending (different object) and transfer-declined (different state).",
    },
    {
        "name": "estimate_transfer_arrival_time",
        "description": "Customer asks how long a transfer will take to arrive or appear, by region/source.",
        "text_ids": [b(x) for x in ["002076", "002043", "002078", "002719"]],
        "reasoning": "Verb=estimate timing, object=transfer arrival. Different from pending-status (which is about diagnosing delay vs general timing expectation).",
    },
    {
        "name": "cancel_transfer",
        "description": "Customer wants to cancel a pending or wrong-amount transfer they initiated.",
        "text_ids": [b(x) for x in ["000703", "000700"]],
        "reasoning": "Verb=cancel, object=transfer. Distinct from cancel_purchase (different object) and cancel_card_payment.",
    },
    {
        "name": "explain_transfer_fee",
        "description": "Customer asks about / questions the fee charged for making a transfer.",
        "text_ids": [b(x) for x in ["002206", "002239"]],
        "reasoning": "Verb=explain, object=transfer fee. Different from card-payment-fee, cash-withdrawal-fee, exchange-fee (different fee objects).",
    },
    {
        "name": "ask_supported_transfer_method",
        "description": "Customer asks whether a specific transfer method/source (e.g. SWIFT) is supported.",
        "text_ids": [b(x) for x in ["000609"]],
        "reasoning": "Verb=ask supported, object=transfer source. Distinct from top-up-method (different action) and how-to-receive (different role).",
    },
    {
        "name": "instructions_send_transfer",
        "description": "Customer asks how to transfer money into their account from a bank or how a friend can send them money.",
        "text_ids": [b(x) for x in ["002326", "002264"]],
        "reasoning": "Verb=instructions/how-to, object=initiating an incoming transfer. Different from top-up methods and from outgoing-transfer questions.",
    },
    {
        "name": "transfer_recipient_visibility",
        "description": "Customer asks how the recipient can see / verify a transfer.",
        "text_ids": [b(x) for x in ["000846"]],
        "reasoning": "Verb=show, object=transfer-on-recipient-side. Distinct from sender-side transfer tracking.",
    },
    {
        "name": "contact_support_about_transfer",
        "description": "Customer asks how to reach customer support specifically regarding a transfer.",
        "text_ids": [b(x) for x in ["001734"]],
        "reasoning": "Verb=contact, object=support. Distinct from any informational question because user is explicitly asking to escalate.",
    },
    {
        "name": "investigate_missing_transfer_to_payee",
        "description": "Customer's payment/transfer didn't reach a specific payee (e.g. landlord, seller); asks to investigate.",
        "text_ids": [b(x) for x in ["000858", "001796"]],
        "reasoning": "Verb=investigate, object=payment that didn't reach payee. Different from generic transfer-pending because user explicitly says recipient denies receiving it.",
    },
    # ---- CARD PAYMENT ----
    {
        "name": "explain_cancelled_card_payment",
        "description": "Customer asks why a card payment was cancelled / didn't go through / contactless payment failed.",
        "text_ids": [b(x) for x in ["002103", "000568", "001625", "001614", "000397"]],
        "reasoning": "Verb=explain, object=cancelled card payment. Different from declined transfer (different object) and broader card-not-working (which is about card status, not a specific payment).",
    },
    {
        "name": "explain_card_payment_fee",
        "description": "Customer asks why a fee was charged for a card payment / why payments are charged at all.",
        "text_ids": [b(x) for x in ["000837", "000833"]],
        "reasoning": "Verb=explain, object=fee on card payment. Different from transfer fee, ATM fee, exchange fee (different fee objects).",
    },
    {
        "name": "explain_pending_card_payment",
        "description": "Customer asks why a card payment is still pending / when it will clear / what pending means.",
        "text_ids": [b(x) for x in ["001609", "001623"]],
        "reasoning": "Verb=explain, object=pending card payment. Different from pending_transfer (different object).",
    },
    {
        "name": "dispute_wrong_charge_amount",
        "description": "Customer was charged an incorrect amount (more or less than expected) for a purchase.",
        "text_ids": [b(x) for x in ["000184", "001689"]],
        "reasoning": "Verb=dispute, object=wrong-amount charge. Different from duplicate charge and unknown charge.",
    },
    {
        "name": "dispute_duplicate_charge",
        "description": "Customer was charged twice for the same transaction and wants the duplicate refunded.",
        "text_ids": [b(x) for x in ["001984", "001998", "001967"]],
        "reasoning": "Verb=dispute, object=duplicate charge. Distinct from refund-not-received and unknown-charge.",
    },
    {
        "name": "dispute_unknown_charge",
        "description": "Customer asks about a specific small/unknown charge they noticed on their statement (e.g. random $1).",
        "text_ids": [b(x) for x in ["000171", "000166", "001484", "001517", "000173", "000172"]],
        "reasoning": "Verb=explain/identify, object=unrecognized charge. Different from fraudulent charge (which alleges another person) and from fee questions (where fee type is known).",
    },
    {
        "name": "dispute_late_charge",
        "description": "Customer noticed a large/unauthorized charge weeks later and asks if they can still dispute it.",
        "text_ids": [b(x) for x in ["001510", "001504"]],
        "reasoning": "Verb=dispute, object=late-noticed charge. Different from immediate unknown-charge dispute (timeliness question) and from duplicate/wrong-amount.",
    },
    {
        "name": "cancel_purchase",
        "description": "Customer wants to cancel a specific purchase or transaction they made.",
        "text_ids": [b(x) for x in ["000710", "001701"]],
        "reasoning": "Verb=cancel, object=purchase/transaction. Different from cancel_transfer (different object).",
    },
    {
        "name": "card_payment_returned",
        "description": "Customer's card payment shows as returned or reversed.",
        "text_ids": [b(x) for x in ["002099"]],
        "reasoning": "Verb=explain, object=returned card payment. Distinct lifecycle event from cancelled-card-payment (return = funds came back).",
    },
    # ---- REFUND ----
    {
        "name": "request_refund_for_unwanted_purchase",
        "description": "Customer wants a refund for an item they didn't want or bought by accident.",
        "text_ids": [b(x) for x in ["001705"]],
        "reasoning": "Verb=request, object=refund for change-of-mind purchase. Different from missing-refund and dispute-charge.",
    },
    {
        "name": "refund_not_received",
        "description": "Customer was promised / requested a refund but it hasn't appeared in their account.",
        "text_ids": [b(x) for x in ["001768", "001799", "001785", "001762"]],
        "reasoning": "Verb=investigate, object=missing refund. Different from request-refund (different stage) and dispute-charge.",
    },
    {
        "name": "dispute_charge_made_by_other",
        "description": "Customer alleges someone else (e.g. ex-partner) made charges on their card and wants refund.",
        "text_ids": [b(x) for x in ["001485"]],
        "reasoning": "Verb=dispute, object=charge by known third party. Different from anonymous-fraud (different attribution) and unknown-charge (knows who).",
    },
    # ---- FRAUD / LOST / STOLEN ----
    {
        "name": "report_fraudulent_charge",
        "description": "Customer reports a charge / transaction / account use they did not make.",
        "text_ids": [b(x) for x in ["001414", "001108", "001435", "001429", "001408"]],
        "reasoning": "Verb=report, object=fraudulent charge/access. Different from unknown-fee (which is about explanation, not fraud) and dispute-by-known-other.",
    },
    {
        "name": "report_stolen_card",
        "description": "Customer reports their card was stolen and asks how to handle it.",
        "text_ids": [b(x) for x in ["000469", "000451", "001672", "000460", "000449"]],
        "reasoning": "Verb=report, object=stolen card. Distinct from lost-wallet/phone and fraudulent-charge.",
    },
    {
        "name": "lost_wallet_or_access",
        "description": "Customer lost wallet or fears card is being misused after loss; requests account protection.",
        "text_ids": [b(x) for x in ["002748", "001644"]],
        "reasoning": "Verb=protect, object=account/funds after wallet loss. Different from stolen-card (different cause) and lost-phone (different device).",
    },
    {
        "name": "report_lost_phone",
        "description": "Customer lost their phone and asks about account / app security implications.",
        "text_ids": [b(x) for x in ["001650", "001669", "001670"]],
        "reasoning": "Verb=report, object=lost phone. Different from lost wallet (different device) and stolen card.",
    },
    # ---- CARD STATUS / ACTIVATION ----
    {
        "name": "activate_card",
        "description": "Customer asks how to activate a new card or when it will be activated.",
        "text_ids": [b(x) for x in ["002865", "002856", "002859", "002866", "002847"]],
        "reasoning": "Verb=activate, object=card. Distinct from card_not_working and from link_card_to_app.",
    },
    {
        "name": "reactivate_card",
        "description": "Customer asks to reactivate / re-enable a previously disabled card.",
        "text_ids": [b(x) for x in ["000041"]],
        "reasoning": "Verb=reactivate, object=card. Distinct from activate (first-time) — explicit re-enable.",
    },
    {
        "name": "unblock_card",
        "description": "Customer asks how to unblock their card.",
        "text_ids": [b(x) for x in ["000538"]],
        "reasoning": "Verb=unblock, object=card. Different from activate/reactivate (different lifecycle state).",
    },
    {
        "name": "card_not_working_general",
        "description": "Customer reports card is generally not working / nothing goes through.",
        "text_ids": [b(x) for x in ["000376", "000371", "000372", "000380"]],
        "reasoning": "Verb=diagnose, object=card-not-working. Different from declined-payment (specific event) and activation.",
    },
    {
        "name": "virtual_card_not_working",
        "description": "Customer's virtual / disposable card isn't working at merchant.",
        "text_ids": [b(x) for x in ["002535", "002542", "002557", "002528", "002526"]],
        "reasoning": "Verb=fix, object=virtual card not working. Different from physical card-not-working (different object) and from disposable-card-usage questions.",
    },
    {
        "name": "card_about_to_expire",
        "description": "Customer notifies that their card is about to expire.",
        "text_ids": [b(x) for x in ["002953"]],
        "reasoning": "Verb=notify/handle, object=expiring card. Distinct because expiry is its own lifecycle event.",
    },
    {
        "name": "link_card_to_app",
        "description": "Customer asks how to sync / register / find their card in the mobile app.",
        "text_ids": [b(x) for x in ["000061", "000074", "000045"]],
        "reasoning": "Verb=link, object=card-to-app. Different from activate-card (app linking ≠ activation in this dataset).",
    },
    # ---- CARD DELIVERY & ORDERING ----
    {
        "name": "ask_card_delivery_eta",
        "description": "Customer asks how long card delivery takes, or whether their card has arrived.",
        "text_ids": [b(x) for x in ["000019", "000022", "000011"]],
        "reasoning": "Verb=estimate, object=card delivery time. Different from expedite (different intent) and tracking (different info).",
    },
    {
        "name": "expedite_card_delivery",
        "description": "Customer asks to receive their card faster or by a specific date.",
        "text_ids": [b(x) for x in ["000285", "000294", "000310"]],
        "reasoning": "Verb=expedite, object=card delivery. Distinct from generic delivery ETA — explicit urgency request.",
    },
    {
        "name": "track_card_shipment",
        "description": "Customer asks for the tracking number for their card shipment.",
        "text_ids": [b(x) for x in ["000039", "000006"]],
        "reasoning": "Verb=track, object=card shipment. Different from ETA question (asks for tracking number specifically).",
    },
    {
        "name": "order_physical_card",
        "description": "Customer asks how to get/order a physical card, or its price, or a second card.",
        "text_ids": [b(x) for x in ["002501", "002484", "002422"]],
        "reasoning": "Verb=order, object=physical card. Different from virtual card request and delivery questions.",
    },
    {
        "name": "order_virtual_card",
        "description": "Customer asks how to purchase / sign up for a virtual card.",
        "text_ids": [b(x) for x in ["000954", "000944"]],
        "reasoning": "Verb=order, object=virtual card. Different from physical-card order (different object) and from disposable-virtual-card usage questions.",
    },
    {
        "name": "ask_card_delivery_eligibility",
        "description": "Customer asks where the card can be delivered / used (e.g. abroad, US, non-UK).",
        "text_ids": [b(x) for x in ["003079", "003048", "003070"]],
        "reasoning": "Verb=check, object=card delivery eligibility. Different from country-support (broader) and from order requests.",
    },
    {
        "name": "ask_supported_card_brands",
        "description": "Customer asks which card brands / networks are supported (Visa, Mastercard, Amex).",
        "text_ids": [b(x) for x in ["001304", "000919", "000889", "001285", "001297"]],
        "reasoning": "Verb=ask supported, object=card brands. Different from country-support and from top-up-methods.",
    },
    # ---- DISPOSABLE CARD ----
    {
        "name": "explain_disposable_card_usage",
        "description": "Customer asks how a disposable virtual card works or what to do with it.",
        "text_ids": [b(x) for x in ["002624", "002600", "002621"]],
        "reasoning": "Verb=explain, object=disposable card usage. Different from disposable-card-limits and virtual-card-not-working.",
    },
    {
        "name": "disposable_card_limits",
        "description": "Customer asks about limits / counts / restrictions on disposable cards.",
        "text_ids": [b(x) for x in ["001393", "001392", "002637"]],
        "reasoning": "Verb=ask limits, object=disposable card. Different from usage explanation and top-up limits.",
    },
    # ---- PIN ----
    {
        "name": "change_pin",
        "description": "Customer asks how / where to change their PIN.",
        "text_ids": [b(x) for x in ["002121", "002151", "002152", "002120"]],
        "reasoning": "Verb=change, object=PIN. Different from reactivate-PIN, unlock-PIN, retrieve-PIN.",
    },
    {
        "name": "unlock_pin",
        "description": "Customer asks how to unlock their PIN after too many failed attempts.",
        "text_ids": [b(x) for x in ["000540"]],
        "reasoning": "Verb=unlock, object=PIN. Different from change/reactivate PIN.",
    },
    {
        "name": "reactivate_pin",
        "description": "Customer asks to reactivate / reinstate their PIN.",
        "text_ids": [b(x) for x in ["000522", "000551"]],
        "reasoning": "Verb=reactivate, object=PIN. Different from unlock (different cause).",
    },
    {
        "name": "retrieve_pin",
        "description": "Customer asks for their PIN or how to check their card's PIN.",
        "text_ids": [b(x) for x in ["001269", "001251"]],
        "reasoning": "Verb=retrieve, object=PIN. Different from change/unlock.",
    },
    # ---- IDENTITY VERIFICATION ----
    {
        "name": "ask_why_verify_identity",
        "description": "Customer asks why they need to verify their identity / what's the point.",
        "text_ids": [b(x) for x in ["001190", "001176", "001172", "001166", "001194"]],
        "reasoning": "Verb=explain, object=identity verification requirement. Different from how-to-verify and verification-failed.",
    },
    {
        "name": "identity_verification_failed",
        "description": "Customer reports they cannot verify their identity / verification is failing.",
        "text_ids": [b(x) for x in ["001220", "001205", "001231", "001232"]],
        "reasoning": "Verb=fix, object=failing identity verification. Different from how-to (different stage) and why (different question).",
    },
    {
        "name": "instructions_verify_identity",
        "description": "Customer asks how to verify their identity, or asks for help with it.",
        "text_ids": [b(x) for x in ["003013", "003030", "003015"]],
        "reasoning": "Verb=instructions, object=identity verification. Different from why (motivation) and failed (error).",
    },
    {
        "name": "ask_identity_documents_needed",
        "description": "Customer asks which documents are required for identity verification.",
        "text_ids": [b(x) for x in ["003028", "003036", "003009"]],
        "reasoning": "Verb=ask documents, object=identity check. Specific docs question; different from broader how-to/why.",
    },
    # ---- TOP UP ----
    {
        "name": "topup_failed_apple_pay",
        "description": "Customer's top-up using Apple Pay / American Express is failing.",
        "text_ids": [b(x) for x in ["002978", "002997", "002986"]],
        "reasoning": "Verb=fix, object=topup-via-applepay. Different from generic topup-failed (different method) and topup-not-showing.",
    },
    {
        "name": "topup_failed_generic",
        "description": "Customer's top-up didn't go through or was denied — generic failure not tied to a specific wallet.",
        "text_ids": [b(x) for x in ["002653", "002642", "002668"]],
        "reasoning": "Verb=diagnose, object=failed top-up. Different from Apple Pay specific and from topup-not-showing.",
    },
    {
        "name": "topup_not_showing_or_delayed",
        "description": "Top-up completed but funds aren't showing / there's a delay.",
        "text_ids": [b(x) for x in ["001003", "001341", "000643"]],
        "reasoning": "Verb=investigate, object=missing top-up funds. Different from failed (which never completed) and verification.",
    },
    {
        "name": "check_topup_status",
        "description": "Customer asks how to tell whether their top-up succeeded or failed.",
        "text_ids": [b(x) for x in ["002676"]],
        "reasoning": "Verb=check, object=topup status. Different from topup-not-showing (which assumes failure).",
    },
    {
        "name": "ask_topup_methods",
        "description": "Customer asks which methods / wallets they can use to top up (Google Pay, Apple Watch, cash, transfer, currency).",
        "text_ids": [b(x) for x in ["002466", "002970", "002989", "002476", "002335", "000880"]],
        "reasoning": "Verb=ask methods, object=top-up. Different from auto-topup, fees, limits, verify.",
    },
    {
        "name": "ask_topup_limit",
        "description": "Customer asks whether there is a top-up limit and what it is.",
        "text_ids": [b(x) for x in ["000756", "000746", "000737", "000726", "000722", "000751"]],
        "reasoning": "Verb=ask limit, object=top-up. Different from auto-topup-limit (different sub-object) and fees.",
    },
    {
        "name": "ask_auto_topup",
        "description": "Customer asks about auto top-up features, options, intervals, or limits.",
        "text_ids": [b(x) for x in ["000321", "000332", "000340", "000351", "000334", "000328"]],
        "reasoning": "Verb=ask, object=auto top-up. Different from regular top-up methods/limits (different feature).",
    },
    {
        "name": "ask_topup_fee",
        "description": "Customer asks about fees for topping up (incl. by transfer or by US card) and Apple Pay cost.",
        "text_ids": [b(x) for x in ["000631", "002827", "002984"]],
        "reasoning": "Verb=ask fee, object=top-up. Different from transfer fee (different action) and exchange fee.",
    },
    {
        "name": "verify_topup",
        "description": "Customer asks how / why to verify a top-up, or where to find the verification code.",
        "text_ids": [b(x) for x in ["002394", "002374", "002381", "002379", "002396"]],
        "reasoning": "Verb=verify, object=top-up. Different from identity verification (different object).",
    },
    # ---- CASH WITHDRAWAL / ATM ----
    {
        "name": "explain_declined_cash_withdrawal",
        "description": "Customer asks why their cash withdrawal was declined / they couldn't take money out.",
        "text_ids": [b(x) for x in ["001596", "001583", "001575"]],
        "reasoning": "Verb=explain, object=declined cash withdrawal. Different from declined transfer (different object) and ATM-stuck-card.",
    },
    {
        "name": "explain_cash_withdrawal_fee",
        "description": "Customer asks why they were charged a fee for a cash withdrawal at an ATM.",
        "text_ids": [b(x) for x in ["002897", "002913", "002904", "002896", "002899", "002915"]],
        "reasoning": "Verb=explain, object=cash withdrawal fee. Different from transfer fee, exchange fee, card-payment fee.",
    },
    {
        "name": "incorrect_cash_dispensed",
        "description": "ATM gave the wrong amount of cash, or the machine ate the money mid-withdrawal.",
        "text_ids": [b(x) for x in ["000789", "000792", "001958", "000218"]],
        "reasoning": "Verb=report, object=incorrect cash. Different from declined withdrawal (transaction failed at request) and ATM-stuck-card (different fault).",
    },
    {
        "name": "atm_stuck_card_or_machine",
        "description": "ATM ate the customer's card or the machine is stuck / broken.",
        "text_ids": [b(x) for x in ["001954", "001926"]],
        "reasoning": "Verb=recover, object=stuck card/ATM. Different from incorrect-cash (different fault).",
    },
    {
        "name": "report_unknown_cash_withdrawal",
        "description": "Customer notices a cash withdrawal on statement they didn't make.",
        "text_ids": [b(x) for x in ["002734"]],
        "reasoning": "Verb=report, object=unknown cash withdrawal. Different from unknown-charge (different transaction type) and fraudulent-charge.",
    },
    {
        "name": "find_atm_for_card",
        "description": "Customer asks which ATMs accept their card / how far the nearest ATM is.",
        "text_ids": [b(x) for x in ["001472", "001461", "001474", "001455"]],
        "reasoning": "Verb=find, object=compatible ATM. Different from change-PIN-ATM (different action).",
    },
    # ---- EXCHANGE ----
    {
        "name": "exchange_rate_incorrect",
        "description": "Customer reports the exchange rate applied to a purchase / cash / card payment was wrong.",
        "text_ids": [b(x) for x in ["000154", "002584", "002572", "002574", "000120", "000157"]],
        "reasoning": "Verb=dispute, object=exchange rate. Different from asking exchange rate (info-only) and fee.",
    },
    {
        "name": "ask_exchange_rate",
        "description": "Customer asks what the exchange rates are.",
        "text_ids": [b(x) for x in ["000080", "000106", "000081"]],
        "reasoning": "Verb=ask, object=exchange rate. Different from disputing the applied rate.",
    },
    {
        "name": "ask_exchange_fee",
        "description": "Customer asks about fees / discounts for exchanging currency / cash.",
        "text_ids": [b(x) for x in ["002571", "002771", "002786", "002788", "002765"]],
        "reasoning": "Verb=ask fee, object=currency exchange. Different from card-payment fee, transfer fee, ATM fee.",
    },
    {
        "name": "exchange_currency_in_app",
        "description": "Customer asks how to exchange currencies in the app / hold multiple currencies.",
        "text_ids": [b(x) for x in ["000417", "000418", "000438", "000273"]],
        "reasoning": "Verb=exchange, object=currency-in-app. Different from rate questions and fees.",
    },
    # ---- ACCOUNT / APP ----
    {
        "name": "open_child_account",
        "description": "Customer asks how to open an account for a child.",
        "text_ids": [b(x) for x in ["000483", "000503"]],
        "reasoning": "Verb=open, object=child account. Different from generic account changes.",
    },
    {
        "name": "close_or_delete_account",
        "description": "Customer wants to close, delete, or get rid of their account.",
        "text_ids": [b(x) for x in ["001892", "001914", "001916"]],
        "reasoning": "Verb=close, object=account. Different from open-account (opposite lifecycle).",
    },
    {
        "name": "change_account_details",
        "description": "Customer asks to change their account details.",
        "text_ids": [b(x) for x in ["001133"]],
        "reasoning": "Verb=change, object=account details. Different from PIN change (different object) and close (different action).",
    },
    # ---- BALANCE / DEPOSIT / SOURCE OF FUNDS ----
    {
        "name": "cheque_deposit_not_credited",
        "description": "Customer deposited a cheque but balance hasn't updated / asks how long it takes.",
        "text_ids": [b(x) for x in ["001066", "001074", "001042"]],
        "reasoning": "Verb=investigate, object=cheque deposit. Different from cash-deposit-not-credited (different deposit type).",
    },
    {
        "name": "cash_deposit_not_credited",
        "description": "Customer's cash deposit hasn't updated in their account.",
        "text_ids": [b(x) for x in ["001058"]],
        "reasoning": "Verb=investigate, object=cash deposit. Different from cheque-deposit (different type).",
    },
    {
        "name": "check_source_of_funds",
        "description": "Customer asks how to check / look up where funds came from (source of funds).",
        "text_ids": [b(x) for x in ["002031", "002035", "002002", "002019", "002021"]],
        "reasoning": "Verb=check source, object=funds. Different from unknown-charge (outflows) and unknown-transaction-origin (similar but transaction-specific).",
    },
    {
        "name": "explain_unknown_transaction",
        "description": "Customer asks about the origin / timing of a specific transaction they don't recognize.",
        "text_ids": [b(x) for x in ["001109"]],
        "reasoning": "Verb=explain, object=unknown transaction origin. Different from source-of-funds (broader) and fraudulent-charge (which alleges someone else's action).",
    },
    {
        "name": "money_disappeared_after_withdrawal",
        "description": "Customer says money was withdrawn but funds are missing or not where expected.",
        "text_ids": [b(x) for x in ["000773"]],
        "reasoning": "Verb=investigate, object=missing-pushed-money. Edge case kept separate from cash-dispensed-incorrect (at ATM) and from top-up-not-showing.",
    },
    # ---- CONTACTLESS ----
    {
        "name": "setup_contactless",
        "description": "Customer asks how to make contactless payments work, including for the metro.",
        "text_ids": [b(x) for x in ["000585", "000577"]],
        "reasoning": "Verb=setup, object=contactless. Payment-method setup, not a card-not-working diagnosis.",
    },
    # ---- SALARY ----
    {
        "name": "ask_salary_deposit",
        "description": "Customer asks if they can receive salary / paycheck through the account, including in different currency.",
        "text_ids": [b(x) for x in ["002240", "002243", "002279", "002274"]],
        "reasoning": "Verb=ask receive, object=salary. Different from generic incoming transfers (specific to salary use case).",
    },
    # ---- COUNTRY / SUPPORT ----
    {
        "name": "ask_supported_countries",
        "description": "Customer asks which countries are supported / where they need to live to use service.",
        "text_ids": [b(x) for x in ["000279", "003051", "003062", "003077"]],
        "reasoning": "Verb=ask supported, object=countries. Different from card-delivery-eligibility (about shipping a card).",
    },
    {
        "name": "ask_card_use_abroad",
        "description": "Customer asks whether their card will work in another country.",
        "text_ids": [b(x) for x in ["000979"]],
        "reasoning": "Verb=ask works, object=card-abroad. Different from country-support (about service eligibility).",
    },
]


def main():
    # Load corpus
    data = json.load(open(TEMP, encoding="utf-8"))
    corpus_ids = {item["id"] for item in data}

    # Validate
    used = []
    for c in clusters:
        used.extend(c["text_ids"])
    used_set = set(used)
    dup = [x for x in used if used.count(x) > 1]
    missing_in_corpus = [x for x in used if x not in corpus_ids]
    unclustered = sorted(corpus_ids - used_set)

    print(f"clusters: {len(clusters)}")
    print(f"total assignments: {len(used)}")
    print(f"unique assignments: {len(used_set)}")
    print(f"duplicates: {sorted(set(dup))[:20]}")
    print(f"missing-in-corpus: {missing_in_corpus[:20]}")
    print(f"unclustered: {len(unclustered)}")

    assert not dup, f"Duplicate text_id assignments: {sorted(set(dup))}"
    assert not missing_in_corpus, f"text_ids not in corpus: {missing_in_corpus}"

    proposal = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_size": len(data),
        "sample_strategy": "random",
        "style": "action_object",
        "existing_clusters_considered": False,
        "clusters": clusters,
        "unclustered_ids": unclustered,
        "observations": (
            "Action-object framing applied throughout: every cluster name is "
            "<verb>_<object>. Banking customer-service intents split fine-grained by both "
            "axes so that, e.g., explain_declined_transfer, explain_declined_cash_withdrawal, "
            "and explain_cancelled_card_payment are distinct clusters (same verb, different "
            "object), and order_physical_card vs activate_card vs reactivate_card vs unblock_card "
            "are distinct (same object, different verb). Same applies for PIN (change/unlock/"
            "reactivate/retrieve), top-up (failed/not-showing/methods/limit/auto/fee/verify), "
            "and refund (request/missing/dispute). Fees are split by what is being charged "
            "(transfer fee, ATM fee, card-payment fee, exchange fee, top-up fee). Pending and "
            "declined states are split by which object they apply to. A few singletons remain "
            "(card_about_to_expire, transfer_recipient_visibility, money_disappeared_after_"
            "withdrawal) — each captures a clear, distinct intent rather than being lumped into "
            "an other bucket."
        ),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(proposal, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {OUT}")
    return len(clusters), len(unclustered)


if __name__ == "__main__":
    main()
