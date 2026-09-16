"""One-off script: write the v2 auditor pass.

Hand-picked assignments are listed in ASSIGNMENTS keyed by sample index (0-based).
Format: (cluster_id, confidence_int_1to5)
"""
import json, os, uuid, datetime, sys

WORKSPACE = os.environ["CLUSTERING_WORKSPACE"]
SAMPLE_PATH = sys.argv[1]

# Index -> (cluster_id, confidence)
# Built from manual reading of the 300-text sample against the 89-cluster v2 taxonomy.
ASSIGNMENTS = {
    0:  ("c32", 4),   # How do I make my virtual card work? -> virtual_card_not_working
    1:  ("c31", 5),   # Fix my contactless -> contactless_not_working
    2:  ("c48", 5),   # change my PIN, only at bank? -> set_or_change_pin
    3:  ("c15", 5),   # app revert a payment -> card_payment_reverted
    4:  ("c56", 4),   # troubleshoot Google pay top up -> topup_failed
    5:  ("c62", 5),   # automatic top-up -> auto_topup
    6:  ("c1", 5),    # Why did I get a transfer declined?
    7:  ("c68", 5),   # Why did I not get all the cash -> atm_wrong_cash_amount
    8:  ("c40", 4),   # Where is the virtual card? -> order_virtual_card (where/how to obtain) — borderline c42
    9:  ("c1", 5),    # decline message during a transfer
    10: ("c83", 5),   # cards in the EU -> open_account_country (or c46) — region question on getting card; closer to c46 supported countries... actually "Can I get one of your cards in the EU" is about availability/eligibility for card delivery -> c38
    11: ("c45", 5),   # Can I get a mastercard? -> apply_for_card
    12: ("c26", 5),   # help with a lost card
    13: ("c54", 5),   # do I need anything for identity check -> how_to_verify_identity
    14: ("c58", 4),   # top-up cancelled -> topup_reverted
    15: ("c60", 5),   # Apple Watch to top up -> topup_methods
    16: ("c41", 5),   # multiple disposable cards -> order_disposable_virtual_card (how to get more)
    17: ("c23", 5),   # unauthorized payment
    18: ("c87", 5),   # address has changed
    19: ("c1", 4),    # transfer 5 times broken -> transfer_declined
    20: ("c60", 4),   # currencies/cards for topping up -> topup_methods
    21: ("c48", 5),   # steps to change card PIN
    22: ("c13", 5),   # card was not accepted -> card_payment_declined
    23: ("c56", 5),   # top up was denied in the app
    24: ("c57", 5),   # why hasn't top up gone through -> topup_pending
    25: ("c4", 4),    # When does money get transferred to my account -> transfer_timing_info
    26: ("c39", 5),   # new card, US resident -> order_physical_card (with eligibility flavor)
    27: ("c70", 5),   # suspicious cash withdraw -> report_unknown_cash_withdrawal
    28: ("c13", 5),   # credit card was declined
    29: ("c39", 5),   # more physical cards
    30: ("c62", 5),   # running out of money, auto top up
    31: ("c64", 5),   # verification code for top-up
    32: ("c22", 5),   # refund on a product
    33: ("c60", 5),   # which cards can I use to top up -> topup_methods
    34: ("c60", 5),   # AmEx to add money -> topup_methods
    35: ("c2", 5),    # transfer worked but pending long time -> transfer_pending
    36: ("c61", 5),   # top-up as much as I want -> topup_limits
    37: ("c23", 5),   # account deducted by seller without approval -> report_fraudulent_transaction
    38: ("c60", 5),   # cheque to top off -> topup_methods
    39: ("c22", 5),   # receive refund for item
    40: ("c36", 5),   # ordered card 2 weeks no delivery
    41: ("c77", 5),   # support any currency -> supported_currencies
    42: ("c27", 5),   # lost phone, don't want others access
    43: ("c30", 4),   # how can I get my physical card to work -> card_not_working_general
    44: ("c7", 5),    # cancel a transfer
    45: ("c70", 4),   # where's accounting for my cash withdrawal -> report_unknown_cash_withdrawal (or c67 pending)
    46: ("c22", 5),   # how do I get refunded
    47: ("c41", 5),   # disposable virtual card as well -> order_disposable_virtual_card
    48: ("c63", 5),   # European bank card for top up, charged? -> topup_fee
    49: ("c1", 5),    # transfer declined
    50: ("c82", 5),   # where has my available money come from -> source_of_funds
    51: ("c5", 5),    # sent money, not logging on recipient side
    52: ("c11", 5),   # transfer to an account not allowed -> transfer_to_beneficiary_blocked
    53: ("c84", 5),   # kids age to use service -> open_account_age
    54: ("c21", 5),   # refund is missing
    55: ("c8", 5),    # transfer charged fees -> transfer_fee_charged
    56: ("c60", 4),   # setup apple pay to use it -> topup_methods (apple pay setup for topping up)
    57: ("c71", 5),   # charge for a cash withdrawal -> cash_withdrawal_fee
    58: ("c52", 4),   # use app without phone -> app_doesnt_recognize_user (or general access) — really "how to access without phone" — closest c52 borderline; could be c27. Lost phone framing absent. I'll go c52
    59: ("c86", 5),   # delete my account -> close_account
    60: ("c71", 5),   # ATM fee on withdrawal -> cash_withdrawal_fee
    61: ("c42", 5),   # what is a disposable virtual card -> disposable_virtual_card_info
    62: ("c9", 5),    # is there a transfer fee -> transfer_fee_policy
    63: ("c89", 5),   # charged extra fee when using a card -> card_payment_fee (NEW)
    64: ("c9", 5),    # Will a transfer incur a fee -> transfer_fee_policy
    65: ("c60", 5),   # top up account with cheque -> topup_methods
    66: ("c22", 5),   # have an item refunded
    67: ("c46", 4),   # use at any establishment accepting Mastercard (acceptance) — actually customer is asserting; treat as acceptance info -> c46
    68: ("c60", 5),   # top up with cheque
    69: ("c20", 5),   # cancel a payment / purchase
    70: ("c79", 5),   # salary eligible -> salary_setup
    71: ("c19", 5),   # extra $1 charge -> unknown_extra_charge
    72: ("c79", 5),   # transfer paycheck -> salary_setup
    73: ("c82", 5),   # history on funds where came from -> source_of_funds
    74: ("c57", 5),   # top up still pending
    75: ("c19", 4),   # unauthorized fee — borderline c23 fraud vs c19 unknown charge; "unauthorized fee" suggests unexpected/extra, fee-shaped -> c19
    76: ("c46", 5),   # countries my card supported in -> card_acceptance_locations
    77: ("c22", 5),   # order never received want money back -> refund_request
    78: ("c87", 5),   # I moved, update details
    79: ("c63", 5),   # charged when used US issued card to add money -> topup_fee (asks about which cards are free)
    80: ("c89", 5),   # Why did I have to pay extra because I paid with card? -> card_payment_fee (NEW)
    81: ("c60", 5),   # how to top up with my card -> topup_methods
    82: ("c44", 5),   # major card payments accepted -> supported_card_brands
    83: ("c84", 5),   # How old do we have to be
    84: ("c8", 5),    # charged a fee when making this transfer
    85: ("c52", 5),   # not being recognized by the app
    86: ("c84", 5),   # kids age
    87: ("c46", 5),   # places that accept my card for payment
    88: ("c1", 5),    # transfer declined
    89: ("c74", 5),   # where do you get exchange rates -> exchange_rate_info
    90: ("c56", 5),   # topped up but app did not accept -> topup_failed
    91: ("c36", 4),   # card still pending, waiting -> card_delivery_status (waiting for delivery)
    92: ("c4", 5),    # how long for transfer to show -> transfer_timing_info
    93: ("c21", 4),   # how long for refund -> refund_not_received (timing of expected refund)
    94: ("c87", 5),   # change of address form -> change_personal_details
    95: ("c27", 5),   # phone lost, prevent app use
    96: ("c6", 5),    # transfer to my account doesn't show -> transfer_not_reflected_in_balance
    97: ("c6", 5),    # balance doesn't reflect transfer
    98: ("c25", 5),   # stolen card, money taken
    99: ("c35", 5),   # shipping to US -> card_delivery_eta
    100: ("c39", 5),  # how to get an actual card
    101: ("c48", 5),  # change PIN but not in country -> set_or_change_pin
    102: ("c13", 5),  # app refused approved payment -> card_payment_declined
    103: ("c43", 5),  # disposable card transactions limit
    104: ("c61", 4),  # limits on auto top-up — c62 auto_topup or c61 topup_limits; "Why are there limits on auto top-up?" specifically about auto-topup limits -> c62
    105: ("c40", 4),  # where is virtual card located -> order_virtual_card (location/how to access)
    106: ("c74", 5),  # international exchange rates
    107: ("c27", 5),  # got mugged can't use app -> lost_phone (phone stolen during mugging)
    108: ("c87", 5),  # change my details
    109: ("c27", 5),  # phone stolen
    110: ("c74", 5),  # check exchange rate applied -> exchange_rate_info
    111: ("c55", 5),  # issues with identity verification -> identity_verification_failed
    112: ("c27", 4),  # phone not with me, can't access app -> lost_phone (proxy: app access without phone)
    113: ("c19", 5),  # €1 fee in statement -> unknown_extra_charge
    114: ("c26", 5),  # what to do if lost card
    115: ("c13", 5),  # credit card declined twice
    116: ("c67", 5),  # ATM cash but app shows pending -> cash_withdrawal_pending
    117: ("c73", 5),  # exchange rate wrong, overcharged -> exchange_rate_wrong
    118: ("c33", 5),  # need new card fees & time -> card_about_to_expire (renewal cost/time)
    119: ("c32", 5),  # virtual card won't work
    120: ("c39", 5),  # procedure for another card -> order_physical_card (additional)
    121: ("c29", 5),  # found card, reactivate
    122: ("c1", 5),   # account transfer failed -> transfer_declined
    123: ("c84", 5),  # youngest to open account -> open_account_age
    124: ("c18", 5),  # charged twice -> duplicate_charge
    125: ("c64", 5),  # can't locate verification code top-up
    126: ("c30", 5),  # card not working
    127: ("c63", 5),  # fee to transfer money from my bank -> topup_fee (transferring FROM bank into account = top-up; "transfer money from my bank" = top-up by bank transfer)
    128: ("c7", 5),   # cancel a transfer
    129: ("c70", 5),  # didn't get cash but app shows transaction -> report_unknown_cash_withdrawal
    130: ("c44", 5),  # Visa and Mastercard accepted -> supported_card_brands
    131: ("c13", 5),  # buying online keeps declining
    132: ("c75", 5),  # extra fee for currency exchange -> exchange_fee
    133: ("c23", 5),  # unauthorized transaction on statement
    134: ("c23", 5),  # someone has removed money from town haven't been to + freeze -> report_fraudulent_transaction (freeze is secondary; primary is unauthorized) — but request is explicit "freeze my account" -> c24. Customer reports fraud AND asks to freeze. Primary action: freeze. -> c24
    135: ("c79", 5),  # use this account to receive my salary
    136: ("c21", 5),  # bought returned money not in account -> refund_not_received
    137: ("c68", 5),  # withdrawal amount isn't right -> atm_wrong_cash_amount
    138: ("c73", 5),  # exchange rate for withdrawal isn't correct
    139: ("c7", 5),   # any way to cancel a transfer
    140: ("c15", 5),  # card payment cancelled -> card_payment_reverted
    141: ("c60", 5),  # how do I transfer money INTO my account -> topup_methods (self-top-up via transfer) — clear v1/v2 boundary test for c10 vs c60
    142: ("c45", 5),  # Visa or Mastercard -> apply_for_card
    143: ("c21", 5),  # refund not on statement -> refund_not_received
    144: ("c55", 5),  # difficulties verifying identity -> identity_verification_failed
    145: ("c74", 5),  # how exchange rates work
    146: ("c61", 5),  # limit to top-up
    147: ("c60", 5),  # top up with apple pay -> topup_methods
    148: ("c13", 5),  # card keeps declining
    149: ("c82", 5),  # check source of funds
    150: ("c19", 5),  # information about €1 fee
    151: ("c67", 5),  # cash withdrawal not yet confirmed -> cash_withdrawal_pending
    152: ("c33", 5),  # new card old one expiring -> card_about_to_expire
    153: ("c21", 5),  # refund isn't showing up
    154: ("c81", 5),  # how to deposit cash -> deposit_methods
    155: ("c64", 5),  # info about verification code
    156: ("c74", 5),  # where do you acquire exchange rate
    157: ("c64", 5),  # verification code for top-up card
    158: ("c35", 5),  # when expect my card -> card_delivery_eta
    159: ("c12", 5),  # SWIFT transfer okay -> swift_transfer_support
    160: ("c57", 5),  # new customer topping up first time, pending half hour -> topup_pending
    161: ("c4", 4),   # how long to post atm withdraw — ATM withdrawal posting time; closest c67 (pending) or c4 (no, that's transfer). No specific ATM-timing cluster. c67 covers stuck-pending; this asks generic how long ATM withdrawals post — fits c67 only loosely. Mark cluster_id null? Actually c67 description says "stuck in pending status or didn't dispense" — they're asking timing not status. -> null
    162: ("c60", 5),  # locations to top up with cash -> topup_methods
    163: ("c31", 5),  # contactless stopped working
    164: ("c48", 5),  # make a new PIN selection -> set_or_change_pin
    165: ("c39", 4),  # more cards, any fees -> order_physical_card (cost q in context of ordering — per c39 description)
    166: ("c32", 5),  # virtual card isn't working
    167: ("c86", 5),  # terminate my account
    168: ("c87", 5),  # change my information
    169: ("c56", 5),  # topped up but didn't complete -> topup_failed
    170: ("c59", 5),  # don't see top up in wallet -> topup_not_showing
    171: ("c23", 4),  # direct debit may not be right — could be fraud (c23) or unknown charge (c19); "may not be right" leans unauthorized/dispute -> c23
    172: ("c12", 5),  # tell me about SWIFT transfers
    173: ("c47", 4),  # local ATM for British pounds + withdrawal charges — primary is ATM locator + fee q. -> c47 (atm_locator) primary
    174: ("c10", 5),  # dual money transferring from one account to another -> transfer_how_to
    175: ("c42", 5),  # how do I use a disposable virtual card -> disposable_virtual_card_info
    176: ("c71", 5),  # charged fee for withdrawing cash -> cash_withdrawal_fee
    177: ("c66", 5),  # ATM transaction cancelled -> cash_withdrawal_failed
    178: ("c56", 5),  # why can't I top up -> topup_failed
    179: ("c19", 5),  # so many fees on statement -> unknown_extra_charge
    180: ("c74", 5),  # what is the exchange rate
    181: ("c33", 5),  # new card current nearly expired
    182: ("c22", 5),  # want a refund for purchase
    183: ("c5", 5),   # transfer hasn't arrived (sender side)
    184: ("c89", 5),  # charged for something bought online, international -> card_payment_fee (international card-payment fee) — could be c75 exchange_fee. "Why did I get charged for something I bought online? Even though it was international, I thought it would be covered." — they're disputing a fee on a card payment that was international. Lean c89 (fee on card payment) over c75 (FX-specific). -> c89
    185: ("c34", 4),  # only own one credit card from USA, accepted? — asking if they can use/link external US card. Could be c34 (link existing) or c44 (supported brands) or c60 (topup methods). Reading carefully: "Will it be accepted?" — likely topup acceptance context. -> c60
    186: ("c50", 5),  # find my card PIN -> retrieve_pin
    187: ("c39", 5),  # second card to daughter -> order_physical_card
    188: ("c70", 5),  # cash withdrawal showing didn't do
    189: ("c46", 4),  # where can I use this card at an ATM -> c47 atm_locator more specific
    190: ("c15", 5),  # money appeared back into account -> card_payment_reverted
    191: ("c19", 5),  # €1 fee for what
    192: ("c24", 5),  # freeze it
    193: ("c37", 5),  # card delivered specific day -> card_delivery_expedite
    194: ("c75", 5),  # fee for exchanging currencies -> exchange_fee
    195: ("c49", 5),  # unblock blocked pin -> unblock_pin
    196: ("c70", 5),  # extra cash withdrawal didn't authorize
    197: ("c69", 5),  # machine kept my card -> atm_swallowed_card
    198: ("c82", 5),  # verify source of my funds
    199: ("c60", 4),  # which cards and what currency supported -> topup_methods (in context, but ambiguous c44/c77/c60). Two-part: cards + currency. -> c60 (most common framing is topup support)
    200: ("c70", 5),  # withdraw I did not make
    201: ("c39", 5),  # may I have another card
    202: ("c47", 5),  # which ATMs can I go to
    203: ("c60", 5),  # how transferring money INTO my account works -> topup_methods (self-fund)
    204: ("c1", 5),   # transfer keeps getting error -> transfer_declined
    205: ("c68", 5),  # ATM didn't give withdraw amount requested
    206: ("c53", 5),  # do not wish to verify identity -> why_verify_identity
    207: ("c48", 5),  # how do I reset my PIN (card PIN reset = change) — actually "reset" implies could be forgot. But the c50 is retrieve_pin (find existing) and c48 is set/change. "Reset" likely = set new value = c48. -> c48
    208: ("c23", 5),  # direct debit don't recognise
    209: ("c30", 5),  # can't get my card to work
    210: ("c85", 5),  # can my children have own account -> open_child_account
    211: ("c66", 5),  # couldn't choose cash at ATM -> cash_withdrawal_failed
    212: ("c63", 5),  # cost to top up US card -> topup_fee
    213: ("c34", 5),  # where to link new card -> link_existing_card
    214: ("c22", 5),  # how to apply for refund
    215: ("c74", 5),  # what do you base exchange rates on
    216: ("c15", 5),  # credit card cancelled a payment -> card_payment_reverted
    217: ("c54", 5),  # what do I need to verify id -> how_to_verify_identity
    218: ("c85", 5),  # can my daughter open an account
    219: ("c13", 5),  # debit card being declined when I have money
    220: ("c89", 5),  # charged a fee after using my card, shouldn't have been -> card_payment_fee
    221: ("c43", 5),  # disposable card cutoff limit -> disposable_card_limits
    222: ("c46", 5),  # does every place accept this card
    223: ("c69", 5),  # ATM didn't return card
    224: ("c19", 4),  # strange payment in statement — could be c23 fraud or c19 unknown — "strange" neutral -> c19
    225: ("c1", 5),   # transfer appeared not to work
    226: ("c24", 5),  # freeze my card, someone used it -> freeze_or_block_card (action)
    227: ("c84", 5),  # is there an age limit
    228: ("c15", 5),  # someone stopped my payment -> card_payment_reverted
    229: ("c1", 5),   # transfer declined
    230: ("c13", 5),  # card being declined online
    231: ("c22", 4),  # refund on extra pound — could be c22 refund or c19 unknown charge. "I would like a refund on the extra pound I was charged" — explicitly requesting a refund. -> c22
    232: ("c48", 5),  # change my card PIN
    233: ("c40", 5),  # where can I obtain virtual card -> order_virtual_card
    234: ("c4", 5),   # wait for US transfer
    235: ("c40", 5),  # possible to get virtual card -> order_virtual_card
    236: ("c29", 5),  # found card, add to app -> reactivate_card (re-enable previously lost) — could be c34 link. "I found my card, can I add it to the app?" found = previously lost — closest c29 reactivate -> c29
    237: ("c45", 5),  # possible to get Visa -> apply_for_card
    238: ("c59", 5),  # where is money I topped off with -> topup_not_showing
    239: ("c55", 5),  # cannot verify identity
    240: ("c57", 5),  # top-up slow to process -> topup_pending
    241: ("c33", 5),  # card about to expire, cost & time
    242: ("c86", 5),  # delete my account
    243: ("c79", 5),  # use this for my salary
    244: ("c14", 4),  # how long to authorise payment -> card_payment_pending (timing of authorization)
    245: ("c18", 5),  # charged twice
    246: ("c34", 5),  # link my card
    247: ("c62", 5),  # set up automatic top-up for travel
    248: ("c65", 5),  # friends access to top up -> third_party_topup
    249: ("c45", 5),  # receive a Visa and MasterCard -> apply_for_card
    250: ("c48", 5),  # change PIN overseas -> set_or_change_pin
    251: ("c33", 5),  # card expiring shortly, cost & time
    252: ("c32", 5),  # disposable virtual card will not work -> virtual_card_not_working
    253: ("c23", 5),  # payment I did not do
    254: ("c64", 5),  # verification code for top-up card
    255: ("c40", 5),  # where will I find my card — virtual card context — borderline c36 delivery. Without "virtual" context, could be physical. But "find my card" suggests not-yet-located. -> c36 (delivery status) or c40. Plain "Where will I find my card?" — likely physical -> c36
    256: ("c66", 5),  # ATM isn't giving out money
    257: ("c68", 5),  # ATM gave less cash than requested
    258: ("c18", 5),  # transaction shows several times -> duplicate_charge
    259: ("c59", 5),  # why can't I see topup in wallet
    260: ("c61", 5),  # amount limit to top-up
    261: ("c4", 5),   # delivery date if transferred urgently to China
    262: ("c41", 5),  # throw away card -> order_disposable_virtual_card
    263: ("c23", 5),  # didn't make this direct debit
    264: ("c56", 5),  # topping up failed, double check
    265: ("c39", 5),  # buy another card -> order_physical_card
    266: ("c47", 5),  # what kind of ATMs can I use
    267: ("c13", 5),  # got new card, payment keeps declining
    268: ("c70", 5),  # someone stole card but card with me + cash withdrawal -> report_unknown_cash_withdrawal (with stolen-card framing — but c25 stolen card. Re-read: "Someone has stolen my card. Even though I have my card with me, someone just made a £500 cash withdrawal." Primary report = unauthorized cash withdrawal (card not actually stolen physically). -> c70
    269: ("c23", 5),  # purchase not by me -> report_fraudulent_transaction
    270: ("c8", 5),   # charged extra fee when transferred money -> transfer_fee_charged
    271: ("c36", 5),  # card has not arrived
    272: ("c12", 5),  # deal with SWIFT transfers
    273: ("c43", 5),  # more than one disposable card -> disposable_card_limits (count limit)
    274: ("c75", 5),  # cost more if currency exchanged -> exchange_fee
    275: ("c76", 5),  # can I exchange to EUR -> currency_exchange_how (or c77 supported_currencies). "Can I exchange to EUR" = can I do this exchange — feature/availability -> c76
    276: ("c14", 5),  # purchases this morning still pending -> card_payment_pending
    277: ("c73", 5),  # exchange rate wrong after purchase
    278: ("c46", 5),  # filling stations accept my card
    279: ("c36", 5),  # ordered card not arrived
    280: ("c87", 5),  # how to update details
    281: ("c53", 4),  # transfers before identity verification — asking whether verification required -> c53 why_verify_identity (informational)
    282: ("c73", 5),  # overcharged, exchange rate wrong
    283: ("c61", 5),  # top-up limits
    284: ("c43", 5),  # virtual cards caps on use -> disposable_card_limits
    285: ("c74", 5),  # exchange rates based on
    286: ("c87", 5),  # update residence details
    287: ("c70", 5),  # withdrawal that isn't mine
    288: ("c63", 5),  # cost to top up by card
    289: ("c21", 5),  # refund missing from statement
    290: ("c63", 5),  # fees for adding money to international card -> topup_fee
    291: ("c27", 5),  # lost phone or stolen
    292: ("c68", 5),  # ATM wrong amount
    293: ("c65", 5),  # how do people send me money -> third_party_topup (others sending to my account) — borderline c10. "How do people send me money?" — from third-party perspective sending TO me. c65 third_party_topup is "can friends top up my account" — close enough. Actually c10 is sending FROM customer. Here customer is RECEIVER. -> c65
    294: ("c42", 5),  # how do disposable cards work
    295: ("c46", 5),  # businesses accept this card
    296: ("c46", 5),  # use card everywhere
    297: ("c86", 5),  # close an account
    298: ("c89", 4),  # extra charge for money withdrawn — "withdrawn" suggests cash withdrawal -> c71 cash_withdrawal_fee, NOT card_payment_fee. Re-read: "Why is there an extra charge for money that was withdrawn?" -> c71
    299: ("c6", 5),   # bank transfer from UK account hasn't appeared (receiving end) -> transfer_not_reflected_in_balance
}

# Manual corrections after re-reading some borderline cases:
ASSIGNMENTS[10] = ("c38", 4)   # "Can I get one of your cards in the EU" -> card_delivery_eligibility
ASSIGNMENTS[58] = ("c52", 3)   # app without phone — low conf, no perfect fit
ASSIGNMENTS[104] = ("c62", 5)  # "Why are there limits on auto top-up?" -> auto_topup feature q (about limits) -> c62
ASSIGNMENTS[134] = ("c24", 5)  # freeze account request primary
ASSIGNMENTS[161] = (None, 2)   # ATM withdrawal posting timing — no good cluster
ASSIGNMENTS[185] = ("c60", 3)  # ambiguous — topup acceptance for foreign card
ASSIGNMENTS[189] = ("c47", 5)  # ATM locator
ASSIGNMENTS[207] = ("c48", 5)  # reset my PIN -> set/change
ASSIGNMENTS[236] = ("c29", 5)  # found card, reactivate
ASSIGNMENTS[255] = ("c36", 4)  # where will I find my card -> delivery status
ASSIGNMENTS[268] = ("c70", 4)  # stolen card framing but actual report is unauthorized cash withdrawal
ASSIGNMENTS[275] = ("c76", 5)  # exchange to EUR -> currency_exchange_how
ASSIGNMENTS[281] = ("c53", 4)  # transfers before identity verification
ASSIGNMENTS[293] = ("c65", 4)  # how do people send me money -> third_party_topup
ASSIGNMENTS[298] = ("c71", 5)  # extra charge for money withdrawn -> cash_withdrawal_fee

# Load sample
with open(SAMPLE_PATH, encoding="utf-8") as f:
    sample = json.load(f)
assert len(sample) == 300, f"Expected 300, got {len(sample)}"
assert len(ASSIGNMENTS) == 300, f"Expected 300 assignments, got {len(ASSIGNMENTS)}"

assignments = []
for i, t in enumerate(sample):
    cid, conf = ASSIGNMENTS[i]
    assignments.append({
        "text_id": t["id"],
        "cluster_id": cid,
        "confidence": int(conf),
    })

audit = {
    "cluster_definitions_version": 2,
    "sample_size": 300,
    "sample_strategy": "random",
    "assignments": assignments,
    "summary": {
        "weak_clusters": [
            {
                "cluster_ids": ["c10", "c60"],
                "issue": "v2 update largely resolves the prior c10/c60 overlap: 'transfer money INTO my account' (text 141, 203) now lands cleanly in c60 topup_methods per the refined description excluding self-fund. c10 still gets one canonical hit (text 174 'dual money transferring from one account to another'). Boundary is now operable; keep as-is."
            },
            {
                "cluster_ids": ["c48", "c50"],
                "issue": "v2 refinement of c48 (set/change including initial setup) vs c50 (find existing PIN) holds up. 'How do I reset my PIN?' (207) still slightly ambiguous — reset language could fit either; the c48 description's 'switching to a chosen value' resolves it. 'How do I find my card PIN?' (186) -> c50 cleanly."
            },
            {
                "cluster_ids": ["c89", "c8", "c63", "c71", "c75"],
                "issue": "NEW c89 card_payment_fee successfully homes the prior orphans: 63 ('charged extra fee using a card'), 80 ('pay extra because I paid with card'), 184 ('charged for international online purchase'), 220 ('fee after using my card'). The c89 explicit cross-references make boundary clean against c8/c63/c71/c75. Watch text 298 'extra charge for money withdrawn' — 'withdrawn' is the c71 signal, not c89."
            },
            {
                "cluster_ids": ["c34", "c60", "c44"],
                "issue": "Text 185 'I only own one other credit card from the USA. Will it be accepted?' remains genuinely ambiguous between topup_methods (c60), supported_card_brands (c44), and link_existing_card (c34). Low-confidence (3). Consider whether 'will my existing foreign card work for top-up?' deserves a sub-cluster of c60 or a tighter c34 description."
            },
            {
                "cluster_ids": ["c27", "c52"],
                "issue": "Text 58 'How do I use the app if I don't have my phone with me?' has no clean home — c52 (app_doesnt_recognize_user) is about login failure not phone absence; c27 (lost_phone) framing doesn't match (phone not lost, just not present). Confidence 3 reflects genuine taxonomy gap."
            }
        ],
        "observations": [
            "Coverage on the v2 taxonomy is essentially complete; only 1 of 300 texts (text 161 'how long does it take to post atm withdraw') has no cluster fit — there is no 'ATM withdrawal timing/posting info' bucket parallel to c4 transfer_timing_info.",
            "v2 changes (added c89; refined c10/c60, c48/c50) resolved the prior audit's two highest-impact boundary issues. Generic card-payment fees, previously homeless across c8/c63/c71/c75, now have a clear c89 home — confidence on those assignments is uniformly 5.",
            "c70 report_unknown_cash_withdrawal continues to absorb a wide range of phrasings including stolen-card-framed reports where the actual complaint is the withdrawal (text 268).",
            "Single low-confidence (1-3) assignments are concentrated on three texts: 58 (no-phone access), 161 (ATM timing), 185 (foreign-card topup eligibility). None of these reflect cluster-quality issues — they're genuine corpus edge cases.",
            "PIN cluster trio (c48/c49/c50) continues to handle the canonical phrasings cleanly post-v2; no observed misroutes between unblock/set/retrieve in this 300-text sample."
        ]
    }
}

# Output path
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
short = uuid.uuid4().hex[:8]
out = os.path.join(WORKSPACE, "audits", f"audit_{ts}_{short}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print("WROTE:", out)
print("ASSIGNED:", len(assignments))
print("NULL:", sum(1 for a in assignments if a["cluster_id"] is None))
