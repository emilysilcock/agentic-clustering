"""Build the auditor JSON from in-line assignments."""
import json, os, uuid, datetime
from pathlib import Path

ws = Path(os.environ["CLUSTERING_WORKSPACE"])
sample_path = sorted((ws / "audits").glob("_audit_sample_*.jsonl"))[-1]
texts = {d["id"]: d["text"] for d in json.load(open(sample_path))}
assert len(texts) == 300, len(texts)

# (text_id, cluster_id_or_None, confidence_1_5, optional_note)
A = [
    ("banking77-test-001482", "c23", 4, None),  # strange direct debit -> fraud
    ("banking77-test-001395", "c43", 5, None),  # disposable card restrictions
    ("banking77-test-001492", "c23", 5, None),  # didn't complete charge, dispute
    ("banking77-test-000565", "c31", 5, None),  # contactless not working
    ("banking77-test-002684", "c6", 5, None),   # balance update after transfer
    ("banking77-test-001311", "c44", 5, None),  # offer Visa or Mastercard - supported brands
    ("banking77-test-000152", "c73", 5, None),  # foreign currency exchange rate incorrect
    ("banking77-test-000563", "c31", 5, None),  # contactless stopped working
    ("banking77-test-002332", "c10", 4, "ambiguous: could be c10 transfer how-to or c60 topup methods; treating 'transfer money to my account' as topup-by-transfer is also valid"),
    ("banking77-test-002826", "c63", 5, None),  # fees for topup with international card
    ("banking77-test-002589", "c73", 4, "mixes exchange-rate question with ATM amount complaint"),
    ("banking77-test-001751", "c13", 2, "very terse 'double check your funds may be declined' - card payment declined-ish"),
    ("banking77-test-001098", "c23", 5, None),  # don't recognise card payment
    ("banking77-test-002038", "c82", 5, None),  # where does this money come from
    ("banking77-test-002677", "c56", 5, None),  # topup failed
    ("banking77-test-000425", "c76", 4, "very terse 'change currency' - currency exchange how"),
    ("banking77-test-001674", "c27", 5, None),  # phone stolen
    ("banking77-test-000742", "c61", 5, None),  # limit on top-ups
    ("banking77-test-000882", "c34", 5, None),  # link existing US card
    ("banking77-test-000239", "c14", 3, "asks what pending means + can I cancel; multi-intent"),
    ("banking77-test-001180", "c53", 5, None),  # why required to do identity check
    ("banking77-test-002137", "c48", 4, "'new pin needs to be set' - change pin"),
    ("banking77-test-000825", "c75", 3, "ambiguous: card fee avoidance, could be c75 exchange_fee or c8/c63"),
    ("banking77-test-000903", "c77", 5, None),  # currencies you accept for topup
    ("banking77-test-001479", "c47", 4, "locations to withdraw money - atm locator vs cash_withdrawal_how"),
    ("banking77-test-001648", "c27", 4, "can I use card if phone stolen"),
    ("banking77-test-002789", "c75", 5, None),  # exchange fee
    ("banking77-test-000874", "c4", 4, "asks 'how long' but says 'has not gotten' - between c4 timing and c5 not received"),
    ("banking77-test-000639", "c12", 5, None),  # swift transfer
    ("banking77-test-001572", "c66", 5, None),  # cash withdrawal declined
    ("banking77-test-000226", "c67", 5, None),  # cash withdrawal pending
    ("banking77-test-000389", "c30", 4, "'I cannot use my card' - card not working general"),
    ("banking77-test-000452", "c25", 5, None),  # card stolen
    ("banking77-test-000871", "c4", 4, "transfer how long + not received yet"),
    ("banking77-test-000481", "c84", 5, None),  # age to open account
    ("banking77-test-002799", "c75", 5, None),  # exchange fee
    ("banking77-test-001957", "c69", 5, None),  # atm swallowed card
    ("banking77-test-000256", "c77", 5, None),  # fiat currencies supported
    ("banking77-test-001378", "c43", 5, None),  # disposable card transaction limit
    ("banking77-test-003045", "c38", 5, None),  # get card outside UK - delivery eligibility
    ("banking77-test-001123", "c87", 5, None),  # change address
    ("banking77-test-002333", "c10", 4, "transfer funds into my account - could be c10 or c60"),
    ("banking77-test-000702", "c20", 5, None),  # cancel transaction
    ("banking77-test-000728", "c61", 5, None),  # topup limit
    ("banking77-test-001679", "c27", 5, None),  # things stolen incl phone
    ("banking77-test-000755", "c61", 5, None),  # topup limit
    ("banking77-test-001071", "c80", 5, None),  # cash deposit not posted
    ("banking77-test-003052", "c83", 5, None),  # do you support EU
    ("banking77-test-001577", "c66", 5, None),  # declined at ATM
    ("banking77-test-002357", "c10", 4, "transfer money into my account"),
    ("banking77-test-000075", "c34", 5, None),  # link recovered card
    ("banking77-test-002700", "c6", 5, None),   # transfer doesn't show in account
    ("banking77-test-002675", "c13", 4, "money from card declined - card payment/withdrawal declined"),
    ("banking77-test-000776", "c68", 5, None),  # ATM gave $20 instead of $100
    ("banking77-test-001794", "c21", 5, None),  # return doesn't show -> refund not received
    ("banking77-test-001898", "c86", 5, None),  # delete account
    ("banking77-test-002203", "c8", 5, None),   # transfer fee charged
    ("banking77-test-000717", "c20", 4, "want to cancel a transaction"),
    ("banking77-test-002730", "c24", 5, None),  # freeze card - strange withdrawal
    ("banking77-test-002746", "c24", 5, None),  # cancel card unauthorized charges
    ("banking77-test-000782", "c68", 5, None),  # got less cash than asked
    ("banking77-test-003035", "c54", 4, "how to verify identity"),
    ("banking77-test-003008", "c54", 5, None),  # how to verify identity
    ("banking77-test-002787", "c75", 4, "travel currency switch cost - exchange fee"),
    ("banking77-test-000243", "c77", 5, None),  # supports multiple currency
    ("banking77-test-000548", "c49", 5, None),  # unblock pin
    ("banking77-test-001573", "c66", 5, None),  # can't get money from ATM
    ("banking77-test-001874", "c6", 4, "money transfer not showing - between c5/c6/c59"),
    ("banking77-test-001837", "c13", 5, None),  # card declined
    ("banking77-test-000482", "c84", 5, None),  # how old
    ("banking77-test-000536", "c49", 5, None),  # blocked pin
    ("banking77-test-002908", "c71", 4, "withdrawal limit/charge - cash_withdrawal_fee"),
    ("banking77-test-001249", "c50", 5, None),  # where to find new pin
    ("banking77-test-001384", "c43", 5, None),  # disposable card transaction limit
    ("banking77-test-000367", "c30", 5, None),  # problem with card - card not working
    ("banking77-test-002755", "c70", 5, None),  # cash I didn't get
    ("banking77-test-001772", "c21", 5, None),  # waiting on refund
    ("banking77-test-000414", "c76", 5, None),  # change to another currency
    ("banking77-test-000220", "c67", 5, None),  # pending cash withdrawal
    ("banking77-test-000284", "c35", 4, "when will card get here - eta vs status"),
    ("banking77-test-002892", "c71", 5, None),  # fee when got cash
    ("banking77-test-003033", "c54", 5, None),  # documentation for identity check
    ("banking77-test-001526", "c51", 4, "wrong with password - reset passcode"),
    ("banking77-test-002533", "c32", 5, None),  # disposable virtual card not work
    ("banking77-test-001630", "c14", 5, None),  # payments pending
    ("banking77-test-000179", "c19", 5, None),  # extra fee on statement
    ("banking77-test-002871", "c28", 5, None),  # activating card
    ("banking77-test-002825", "c63", 5, None),  # topup fee
    ("banking77-test-000347", "c62", 5, None),  # auto topup
    ("banking77-test-000525", "c49", 5, None),  # wrong pin too many times
    ("banking77-test-002289", "c1", 4, "transfer didn't go through - declined"),
    ("banking77-test-001910", "c86", 5, None),  # get rid of account
    ("banking77-test-000342", "c62", 5, None),  # auto topup option
    ("banking77-test-000270", "c77", 5, None),  # what currency hold money
    ("banking77-test-002013", "c82", 5, None),  # source of funds verify
    ("banking77-test-002316", "c1", 5, None),  # transfer not go through
    ("banking77-test-002090", "c13", 4, "card payment cancelled by merchant/system - card_payment_declined"),
    ("banking77-test-001323", "c10", 5, None),  # use credit card to transfer
    ("banking77-test-001989", "c18", 5, None),  # charged twice
    ("banking77-test-002670", "c56", 4, "app won't let me top up - topup failed"),
    ("banking77-test-001015", "c58", 5, None),  # topup reverted
    ("banking77-test-000783", "c68", 5, None),  # full amount not dispensed
    ("banking77-test-000803", "c19", 4, "extra charge from card use - unknown_extra_charge or c17"),
    ("banking77-test-002104", "c15", 5, None),  # card payment reverted
    ("banking77-test-001094", "c23", 5, None),  # payment I didn't do
    ("banking77-test-000862", "c5", 4, "transfer sent friend hasn't received + how long"),
    ("banking77-test-001433", "c23", 4, "card numbers compromised - fraud-ish; could be freeze c24 too"),
    ("banking77-test-002143", "c48", 5, None),  # change pin at ATM
    ("banking77-test-001043", "c80", 5, None),  # deposit not showing
    ("banking77-test-001252", "c50", 3, "very terse 'do i need a pin' - retrieve_pin loose match"),
    ("banking77-test-001412", "c24", 4, "stop fraud now - freeze/block"),
    ("banking77-test-001754", "c13", 5, None),  # online declined
    ("banking77-test-002482", "c39", 4, "charge for physical cards - c39 order vs c45 apply"),
    ("banking77-test-002515", "c38", 5, None),  # where can cards be delivered
    ("banking77-test-001026", "c58", 5, None),  # topup reverted
    ("banking77-test-000071", "c34", 5, None),  # use old card with app
    ("banking77-test-002320", "c60", 4, "bank transfer to top up account"),
    ("banking77-test-001515", "c23", 3, "direct debit not set up + would like to - ambiguous fraud vs setup"),
    ("banking77-test-002873", "c28", 5, None),  # new card activation
    ("banking77-test-000003", "c35", 5, None),  # when card will arrive
    ("banking77-test-001373", "c43", 5, None),  # multiple disposable cards payments
    ("banking77-test-000209", "c67", 4, "withdrawal show timing - pending"),
    ("banking77-test-000264", "c76", 5, None),  # exchange currencies
    ("banking77-test-000177", "c19", 5, None),  # extra 1 pound charge
    ("banking77-test-000261", "c77", 4, "exchange EUR - supported currencies"),
    ("banking77-test-002988", "c56", 4, "Amex top up not working - topup failed"),
    ("banking77-test-002737", "c70", 5, None),  # weird withdrawal
    ("banking77-test-002928", "c33", 5, None),  # card expiring soon
    ("banking77-test-000616", "c63", 5, None),  # transfer to topup - charged?
    ("banking77-test-000847", "c5", 5, None),  # recipient not received
    ("banking77-test-002160", "c1", 5, None),  # transfer not going through
    ("banking77-test-001920", "c69", 5, None),  # ATM stole card
    ("banking77-test-000169", "c19", 5, None),  # don't remember 1 pound purchase
    ("banking77-test-002362", "c64", 5, None),  # verify topup code
    ("banking77-test-002278", "c79", 4, "paid in another currency - salary setup"),
    ("banking77-test-000095", "c74", 5, None),  # list of exchange rates
    ("banking77-test-000014", "c36", 5, None),  # waiting 1 week for card - delivery status
    ("banking77-test-001552", "c51", 5, None),  # reset passcode
    ("banking77-test-001872", "c2", 5, None),  # how long transfer pending
    ("banking77-test-000993", "c46", 5, None),  # places accept card
    ("banking77-test-002611", "c41", 5, None),  # disposable virtual card
    ("banking77-test-000024", "c36", 5, None),  # card not arrived where is it
    ("banking77-test-002398", "c64", 5, None),  # verification code topup
    ("banking77-test-001237", "c55", 5, None),  # can't verify identity
    ("banking77-test-002634", "c41", 5, None),  # order disposable virtual card
    ("banking77-test-001497", "c23", 5, None),  # suspicious direct debit
    ("banking77-test-000972", "c46", 5, None),  # places use card
    ("banking77-test-002268", "c79", 5, None),  # deposit salary
    ("banking77-test-001600", "c14", 5, None),  # card payment on hold
    ("banking77-test-002402", "c39", 4, "order another card - assume physical"),
    ("banking77-test-001275", "c48", 3, "'what do I need to do for a PIN' - very vague, change/set"),
    ("banking77-test-002491", "c38", 5, None),  # ship cards delivery eligibility
    ("banking77-test-000749", "c61", 5, None),  # topup limit at a time
    ("banking77-test-002594", "c73", 5, None),  # exchange rate abroad
    ("banking77-test-002050", "c4", 4, "how long until money in account - transfer timing"),
    ("banking77-test-002444", "c60", 5, None),  # topup by cash deposit
    ("banking77-test-002359", "c60", 5, None),  # bank transfer refill - topup methods
    ("banking77-test-000466", "c26", 5, None),  # missing card - lost
    ("banking77-test-002197", "c78", 5, None),  # crypto exchange not working
    ("banking77-test-001513", "c23", 5, None),  # direct debit didn't make
    ("banking77-test-001096", "c23", 5, None),  # don't recognize transactions, freeze
    ("banking77-test-001682", "c22", 4, "get item refund - refund request"),
    ("banking77-test-000542", "c52", 5, None),  # account blocked can't log in
    ("banking77-test-001188", "c53", 5, None),  # skip identity verification
    ("banking77-test-001049", "c80", 5, None),  # cheque deposit not credited
    ("banking77-test-002575", "c71", 5, None),  # ATM withdrawal cost
    ("banking77-test-002512", "c39", 5, None),  # real-life card
    ("banking77-test-001949", "c69", 5, None),  # card stuck in ATM
    ("banking77-test-001446", "c47", 5, None),  # which ATMs accept card -> atm locator-ish; also c46
    ("banking77-test-000855", "c5", 5, None),  # recipient doesn't see transfer
    ("banking77-test-001278", "c50", 5, None),  # what is my pin - retrieve pin
    ("banking77-test-001057", "c80", 5, None),  # check deposit balance same
    ("banking77-test-000868", "c16", 4, "transaction taking so long - card_payment_slow"),
    ("banking77-test-001114", "c17", 4, "card payments look different than purchased - wrong amount/place"),
    ("banking77-test-001470", "c72", 4, "where can I withdraw money - cash withdrawal how"),
    ("banking77-test-000747", "c61", 5, None),  # topup limit
    ("banking77-test-001731", "c1", 5, None),  # transfer declined despite details correct
    ("banking77-test-002681", "c6", 5, None),  # balance didn't change after transfer
    ("banking77-test-000527", "c48", 3, "reset PIN can't use card - between c48 change and c49 unblock"),
    ("banking77-test-000094", "c74", 5, None),  # exchange rate on app
    ("banking77-test-001776", "c21", 5, None),  # refund not received
    ("banking77-test-000544", "c49", 5, None),  # unblock pin
    ("banking77-test-001365", "c43", 5, None),  # disposable cards limit
    ("banking77-test-001478", "c46", 4, "places where can't withdraw - card acceptance"),
    ("banking77-test-002056", "c4", 5, None),  # how long transfer takes
    ("banking77-test-000194", "c19", 5, None),  # $1 extra charged
    ("banking77-test-001186", "c55", 4, "when can I use account since id not verified"),
    ("banking77-test-001930", "c69", 5, None),  # ATM won't give card back
    ("banking77-test-001427", "c24", 5, None),  # freeze card unauthorized
    ("banking77-test-002957", "c33", 5, None),  # card expire order new
    ("banking77-test-000707", "c20", 4, "want to go back on what I did - cancel"),
    ("banking77-test-001305", "c45", 5, None),  # need visa and mastercard - apply
    ("banking77-test-001806", "c13", 5, None),  # why decline payment
    ("banking77-test-000866", "c14", 4, "waiting for transaction - pending"),
    ("banking77-test-001265", "c50", 5, None),  # where is pin number
    ("banking77-test-001538", "c51", 5, None),  # reset passcode
    ("banking77-test-002345", "c10", 5, None),  # bank account transfer how
    ("banking77-test-001773", "c21", 5, None),  # refund missing
    ("banking77-test-001105", "c24", 5, None),  # payments didn't make, freeze
    ("banking77-test-002820", "c63", 5, None),  # European bank card topup fee
    ("banking77-test-000595", "c31", 4, "how to use contactless"),
    ("banking77-test-003053", "c38", 5, None),  # card if in USA
    ("banking77-test-000906", "c44", 4, "all cards and currencies - supported brands+currencies"),
    ("banking77-test-000828", "c19", 4, "first time fee - extra fee inquiry"),
    ("banking77-test-000731", "c61", 5, None),  # topup limit
    ("banking77-test-001897", "c86", 5, None),  # terminate account
    ("banking77-test-002793", "c75", 5, None),  # discount frequent exchange - exchange fee
    ("banking77-test-001223", "c52", 5, None),  # app doesn't believe I am me
    ("banking77-test-002167", "c1", 5, None),  # transfer not possible
    ("banking77-test-002867", "c28", 5, None),  # activate new card
    ("banking77-test-001234", "c55", 5, None),  # trouble verifying id
    ("banking77-test-001214", "c55", 5, None),  # id not being verified
    ("banking77-test-000296", "c37", 5, None),  # delivered on certain date - expedite
    ("banking77-test-000958", "c40", 5, None),  # obtain virtual card
    ("banking77-test-002245", "c79", 5, None),  # salary in GBP
    ("banking77-test-001977", "c18", 5, None),  # duplicate charges
    ("banking77-test-002669", "c56", 5, None),  # topup not working
    ("banking77-test-000029", "c36", 5, None),  # track card - delivery status
    ("banking77-test-001250", "c50", 5, None),  # where in app find PIN
    ("banking77-test-000857", "c5", 4, "why transfer didn't get there - recipient"),
    ("banking77-test-000087", "c74", 5, None),  # exchange rate
    ("banking77-test-001964", "c18", 4, "I didn't buy this twice - duplicate charge"),
    ("banking77-test-002923", "c33", 5, None),  # card expires
    ("banking77-test-002828", "c63", 5, None),  # European card topup charge
    ("banking77-test-003002", "c54", 5, None),  # how to verify identity
    ("banking77-test-002219", "c8", 5, None),  # extra fee for transferring
    ("banking77-test-000911", "c60", 4, "cards supported for topup - topup methods"),
    ("banking77-test-000560", "c31", 5, None),  # contactless enabled new card
    ("banking77-test-000799", "c66", 4, "declined withdraw funds returned - cash_withdrawal_failed"),
    ("banking77-test-001676", "c27", 5, None),  # can't find phone with card info
    ("banking77-test-000359", "c62", 5, None),  # enable auto topup
    ("banking77-test-000946", "c42", 4, "where to find virtual card - disposable_virtual_card_info or c40"),
    ("banking77-test-000056", "c34", 5, None),  # link card already have
    ("banking77-test-000399", "c29", 5, None),  # deactivated card not working - reactivate
    ("banking77-test-001590", "c66", 5, None),  # withdraw declined
    ("banking77-test-001759", "c1", 5, None),  # transfer declined
    ("banking77-test-002667", "c56", 5, None),  # topup denied
    ("banking77-test-000390", "c30", 5, None),  # card broken
    ("banking77-test-001867", "c59", 4, "money transferred to me doesn't show - topup_not_showing or c6"),
    ("banking77-test-002536", "c32", 5, None),  # virtual card transactions
    ("banking77-test-002330", "c10", 5, None),  # how to transfer money
    ("banking77-test-000678", "c57", 5, None),  # topup pending
    ("banking77-test-000422", "c76", 5, None),  # exchanging currencies on app
    ("banking77-test-003044", "c38", 5, None),  # in US can I get a card
    ("banking77-test-001419", "c23", 5, None),  # random purchases - hacked
    ("banking77-test-001449", "c46", 4, "which ATMs accept card - card acceptance"),
    ("banking77-test-002128", "c48", 4, "set new PIN - change pin"),
    ("banking77-test-002298", "c1", 5, None),  # transfer keeps failing/rejected
    ("banking77-test-001703", "c22", 5, None),  # item refunded
    ("banking77-test-002582", "c73", 5, None),  # error exchange rate cash withdrawal
    ("banking77-test-002864", "c28", 5, None),  # activating card
    ("banking77-test-001030", "c58", 4, "topup cancelled - reverted"),
    ("banking77-test-002691", "c4", 5, None),  # UK transfer timing
    ("banking77-test-000421", "c76", 4, "need GBP instead of AUD - how to exchange"),
    ("banking77-test-003025", "c54", 5, None),  # where verify identity
    ("banking77-test-001777", "c21", 5, None),  # recent refund not on statement
    ("banking77-test-001405", "c24", 5, None),  # freeze account hacked
    ("banking77-test-000925", "c40", 4, "virtual card not showing - order_virtual_card"),
    ("banking77-test-000311", "c35", 5, None),  # expect to receive new card
    ("banking77-test-001574", "c66", 5, None),  # ATM declining card
    ("banking77-test-000416", "c46", 4, "use money in different country - card acceptance"),
    ("banking77-test-001571", "c66", 5, None),  # ATM declining card
    ("banking77-test-002808", "c63", 5, None),  # European card topup charge
    ("banking77-test-000760", "c68", 5, None),  # ATM gave 10 of 30 pounds
    ("banking77-test-002297", "c1", 5, None),  # transfer fails
    ("banking77-test-001817", "c13", 5, None),  # card payment did not work
    ("banking77-test-002742", "c70", 5, None),  # unauthorized cash withdrawal 500
    ("banking77-test-002569", "c73", 5, None),  # wrong exchange rate abroad
    ("banking77-test-002985", "c60", 4, "topup with Google Pay - topup methods"),
    ("banking77-test-001345", "c10", 5, None),  # credit card to transfer
    ("banking77-test-002711", "c2", 5, None),  # transfer pending
    ("banking77-test-001545", "c51", 5, None),  # forgot password
    ("banking77-test-001044", "c80", 5, None),  # cash deposit not there
    ("banking77-test-000826", "c8", 3, "fee when pay with card - ambiguous c8 transfer fee vs c71 cash fee"),
    ("banking77-test-002657", "c56", 5, None),  # funding card didn't go through
    ("banking77-test-000214", "c67", 5, None),  # waiting for cash withdrawal to show
    ("banking77-test-000931", "c40", 5, None),  # not get virtual card
    ("banking77-test-000158", "c73", 5, None),  # wrong rate applied currency
    ("banking77-test-001836", "c13", 5, None),  # card payment declined
    ("banking77-test-000253", "c77", 5, None),  # fiat currencies supported
    ("banking77-test-000778", "c68", 5, None),  # ATM wrong amount
    ("banking77-test-001139", "c87", 5, None),  # modify my details
    ("banking77-test-002689", "c6", 5, None),  # transfer balance didn't update
    ("banking77-test-002834", "c63", 5, None),  # fee for top ups
    ("banking77-test-002615", "c41", 5, None),  # get disposable virtual card
    ("banking77-test-001206", "c52", 5, None),  # app doesn't know it's me
    ("banking77-test-000183", "c19", 5, None),  # overcharged a pound
    ("banking77-test-001101", "c23", 5, None),  # fraudulent charge
    ("banking77-test-002069", "c4", 5, None),  # funds transfer how long bank to bank
    ("banking77-test-001653", "c27", 5, None),  # left phone at hotel
    ("banking77-test-000835", "c19", 4, "fee on account why - extra charge"),
    ("banking77-test-002017", "c82", 5, None),  # where funds came from
    ("banking77-test-003058", "c83", 3, "'where can I find your locations' - very ambiguous; could be branches/atms/countries"),
    ("banking77-test-000031", "c36", 5, None),  # expecting card not received
    ("banking77-test-001523", "c51", 4, "passcode not working"),
    ("banking77-test-001436", "c23", 5, None),  # card numbers copied
    ("banking77-test-002030", "c82", 5, None),  # see where money comes from
    ("banking77-test-002498", "c39", 4, "charge for physical card - c39"),
    ("banking77-test-002981", "c60", 4, "top up Google pay - methods"),
    ("banking77-test-001377", "c43", 5, None),  # disposable cards limit
]

assert len(A) == 300, f"expected 300, got {len(A)}"
ids_seen = {a[0] for a in A}
missing = set(texts.keys()) - ids_seen
extra = ids_seen - set(texts.keys())
assert not missing, f"missing: {missing}"
assert not extra, f"extra: {extra}"

# Build assignments
assignments = []
for tid, cid, conf, note in A:
    assert isinstance(conf, int) and 1 <= conf <= 5, (tid, conf)
    item = {
        "text_id": tid,
        "cluster_id": cid,
        "confidence": conf,
    }
    if note:
        item["note"] = note
    assignments.append(item)

# Build summary - QUALITATIVE ONLY
# weak_clusters: any cluster where I gave confidence <=3 OR noted ambiguity, or where
# multiple texts collide with sibling clusters
from collections import Counter, defaultdict
low_conf_by_cluster = defaultdict(list)
for tid, cid, conf, note in A:
    if conf <= 3 and cid is not None:
        low_conf_by_cluster[cid].append((tid, conf, note))

weak_clusters = []
# clusters with multiple low-conf hits or repeated boundary issues
boundary_pairs_seen = {
    "c10/c60": "transfer-into-own-account framed as 'transfer money to my account' — overlaps with topup_methods; several texts hit this ambiguity",
    "c8/c63/c71/c75": "fee-with-card questions are split across transfer_fee / topup_fee / cash_withdrawal_fee / exchange_fee; bare 'fee on my card' has no clear home",
    "c47/c72": "'where to withdraw money' ambiguous between atm_locator and cash_withdrawal_how",
    "c46/c47": "'which ATMs accept my card' splits between card_acceptance_locations and atm_locator",
    "c48/c49/c50": "PIN cluster boundaries blur: 'I need to set my PIN' could be activate/change/retrieve",
    "c5/c6/c59": "'transfer not showing' boundary between recipient-not-received, balance-not-updated, and topup-not-showing when third-party sends money in",
    "c40/c42": "'where is my virtual card' could be order_virtual_card (didn't arrive) or disposable_virtual_card_info (how it works)",
    "c39/c45": "'how to get a physical card' overlaps with apply_for_card (specific brand)",
    "c33/c39": "'card expiring, do I need to order a new one' touches both card_about_to_expire and order_physical_card",
}

weak_clusters = [
    {
        "cluster_ids": ["c10", "c60"],
        "issue": "Customers say 'transfer money to my account' meaning top-up-via-bank-transfer; transfer_how_to and topup_methods overlap. Recommend either merging or sharpening the description boundary.",
    },
    {
        "cluster_ids": ["c8", "c63", "c71", "c75"],
        "issue": "Generic fee questions ('there's a fee on my card, why', 'do you charge a fee') have no home — the four fee clusters all require a specific operation context. A general 'fee_explanation' or 'fee_inquiry' cluster (or routing to c19 unknown_extra_charge) would help.",
    },
    {
        "cluster_ids": ["c46", "c47", "c72"],
        "issue": "'Where can I withdraw / which ATMs work' splits across card_acceptance_locations / atm_locator / cash_withdrawal_how with no clear rule. Boundary descriptions need tightening.",
    },
    {
        "cluster_ids": ["c48", "c49", "c50"],
        "issue": "PIN intents (change / unblock / retrieve / set-new-pin) blur on bare phrasings like 'I need to set my PIN' or 'what do I do for a PIN'. c48 should explicitly cover 'setting a PIN for the first time' vs c50 'finding the PIN you were issued'.",
    },
    {
        "cluster_ids": ["c5", "c6", "c59"],
        "issue": "'Transfer not showing' framing splits between transfer_not_received_by_recipient (outgoing), transfer_not_reflected_in_balance (own balance), and topup_not_showing (incoming third-party). When a customer says 'someone transferred me money and it doesn't show', it sits between c6 and c59.",
    },
    {
        "cluster_ids": ["c40", "c42"],
        "issue": "order_virtual_card vs disposable_virtual_card_info conflict on 'where is my virtual card' — could be 'I ordered one and it hasn't arrived' or 'I don't know where to find the feature'.",
    },
]

observations = [
    "Coverage on this 300-text random sample is essentially complete with 88 fine-grained intent clusters; no text required cluster_id=null.",
    "Most low-confidence (<=3) calls cluster around (a) which fee-cluster a bare 'fee' question belongs to and (b) the PIN trio (c48/c49/c50). These look like description-boundary issues rather than missing clusters.",
    "A few texts (e.g. 'change currency', 'do I need a pin', 'where can I find your locations') are so terse that any system would struggle — these are sample-noise, not taxonomy gaps.",
    "The transfer / topup / refund / fraud spines are all well-populated; no large clusters look over-broad. The taxonomy successfully separates very fine distinctions (e.g. transfer_declined vs transfer_pending vs transfer_speed_up).",
    "Recommend tightening the descriptions for the fee clusters (c8/c63/c71/c75) and the PIN clusters (c48/c49/c50) before another audit pass.",
]

summary = {
    "weak_clusters": weak_clusters,
    "observations": observations,
}

audit = {
    "cluster_definitions_version": 1,
    "sample_size": len(assignments),
    "sample_strategy": "random",
    "assignments": assignments,
    "summary": summary,
}

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
short = uuid.uuid4().hex[:8]
out = ws / "audits" / f"audit_{ts}_{short}.json"
out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(f"WROTE: {out}")
print(f"assignments: {len(assignments)}")
print(f"null cluster_id count: {sum(1 for a in assignments if a['cluster_id'] is None)}")
