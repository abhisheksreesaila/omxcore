# F-153: `Query.join` (the builder's inner equi-join) against a plain double loop over the same tables, on the key rules the engine promises:
# NULL keys never match, TEXT keys match by bytes across dictionaries, DECIMAL scales are aligned, INT is scale 0, a repeated key multiplies rows,
# the join comes first.
from omxcore import Table, SharedTable, INT, DECIMAL, TEXT, Query, col, sum, count


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def main() raises:
    # left: id, k (INT, every 7th NULL), tag (TEXT: West / west / East), amount (DECIMAL 2)
    var lt = Table.empty()
    var l_id = lt.add_column("id", INT)
    var l_k = lt.add_column("k", INT)
    var l_tag = lt.add_column("tag", TEXT)
    var l_amt = lt.add_column("amount", DECIMAL, scale=2)
    var l_words: List[String] = ["West", "west", "East", "West "]
    for w in l_words:
        _ = lt.add_word(l_tag, w)
    var n_left = 5000
    for i in range(n_left):
        lt.append(l_id, Int64(i))
        if i % 7 == 6:
            lt.append_null(l_k)
        else:
            lt.append(l_k, Int64((i * 31) % 400))
        lt.append(l_tag, Int64(i % 4))
        lt.append(l_amt, Int64((i % 90) * 25))
    # right: k (INT; 300 distinct, some repeated), kk (DECIMAL 4: the same numbers at another scale), tag (TEXT with a different dictionary), w
    var rt = Table.empty()
    var r_k = rt.add_column("k", INT)
    var r_kk = rt.add_column("kk", DECIMAL, scale=4)
    var r_tag = rt.add_column("tag", TEXT)
    var r_w = rt.add_column("w", INT)
    var r_words: List[String] = ["East", "West", "north"]
    for w in r_words:
        _ = rt.add_word(r_tag, w)
    var n_right = 700
    for i in range(n_right):
        var key = (i * 13) % 300
        rt.append(r_k, Int64(key))
        rt.append(r_kk, Int64(key) * 10000)
        rt.append(r_tag, Int64(i % 3))
        rt.append(r_w, Int64(i))
    var left = lt^.share()
    var right = rt^.share()

    # by hand: pairs on k
    var pairs = 0
    var sum_id = 0
    var sum_w = 0
    for i in range(n_left):
        if i % 7 == 6:
            continue
        var k = (i * 31) % 400
        for j in range(n_right):
            if (j * 13) % 300 == k:
                pairs += 1
                sum_id += i
                sum_w += j
    expect(pairs > 1000, "the test data join: " + String(pairs))
    var r1 = Query(left).join(right, ["k"], ["k"]).agg(count(), sum(col("id")), sum(col("w"))).compute()
    expect(r1.counts[0] == pairs, "count of the join: " + String(r1.counts[0]) + " wanted " + String(pairs))
    expect(r1.values(1)[0] == sum_id and r1.values(2)[0] == sum_w, "sums of the join")
    # the same through DECIMAL scales: k (scale 0) against kk (scale 4)
    var r2 = Query(left).join(right, ["k"], ["kk"]).agg(count()).compute()
    expect(r2.counts[0] == pairs, "INT against DECIMAL(4): " + String(r2.counts[0]))
    # two keys, one of them TEXT with different dictionaries: 'West' matches 'West' only ('west' and 'West ' are other words), 'East' matches 'East'
    var by_hand = 0
    for i in range(n_left):
        if i % 7 == 6:
            continue
        var k = (i * 31) % 400
        var lw = i % 4                                           # 0 West, 1 west, 2 East, 3 'West '
        for j in range(n_right):
            if (j * 13) % 300 == k:
                var rw = j % 3                                   # 0 East, 1 West, 2 north
                if (lw == 0 and rw == 1) or (lw == 2 and rw == 0):
                    by_hand += 1
    var r3 = Query(left).join(right, ["k", "tag"], ["k", "tag"]).agg(count()).compute()
    expect(r3.counts[0] == by_hand, "INT and TEXT keys: " + String(r3.counts[0]) + " wanted " + String(by_hand))
    # the joined table's columns: this table's, then the other's, a repeated name suffixed
    var joined = Query(left).join(right, ["k"], ["k"])
    expect(joined.table.columns() == 8 and joined.table.name(4) == "k_1" and joined.table.name(5) == "kk" and joined.table.name(6) == "tag_1" and joined.table.rows() == 0, "column names")
    expect(joined.joined_table().rows() == pairs and joined.joined_table().name(4) == "k_1", "rows of the joined table")
    # the builder's rules after the join: a dense group key
    var r4 = Query(left).join(right, ["k"], ["k"]).group_by(col("w") % 5, 5).agg(count()).compute()
    var total = 0
    for g in range(5):
        total += r4.counts[g]
    expect(total == pairs, "group_by after the join covers every row")
    # refusals by name
    var refused = False
    try:
        _ = Query(left).join(right, ["k"], ["k"], "full")
    except e:
        refused = "a `full` join is not supported" in String(e)
    expect(refused, "a full join is refused by name")
    # left: every left row; the right's columns are NULL where nothing matches. By hand: the NULL keys and the keys without a partner give one row each
    var no_partner = 0
    var null_keys = 0
    for i in range(n_left):
        if i % 7 == 6:
            null_keys += 1
            continue
        var k = (i * 31) % 400
        var has = False
        for j in range(n_right):
            if (j * 13) % 300 == k:
                has = True
                break
        if not has:
            no_partner += 1
    var lj = Query(left).join(right, ["k"], ["k"], "left")
    expect(lj.joined_table().rows() == pairs + no_partner + null_keys, "left join rows: " + String(lj.joined_table().rows()) + " wanted " + String(pairs + no_partner + null_keys))
    expect(lj.joined_table().null_count(5) == no_partner + null_keys, "the right's columns are NULL where nothing matches: " + String(lj.joined_table().null_count(5)))
    var rl = Query(left).join(right, ["k"], ["k"], "left").agg(count(), sum(col("id"))).compute()
    expect(rl.counts[0] == pairs + no_partner + null_keys, "a left join counted")
    # semi and anti: this table's columns only; a NULL key has no partner (so anti keeps it)
    var sj = Query(left).join(right, ["k"], ["k"], "semi")
    var aj = Query(left).join(right, ["k"], ["k"], "anti")
    expect(sj.table.columns() == 4 and aj.table.columns() == 4, "semi and anti keep this table's columns")
    expect(aj.joined_table().rows() == no_partner + null_keys, "anti rows: " + String(aj.joined_table().rows()))
    expect(sj.joined_table().rows() + aj.joined_table().rows() == n_left, "semi and anti split the rows")
    refused = False
    try:
        _ = Query(left).where(col("id") > 3).join(right, ["k"], ["k"])
    except e:
        refused = "join must come first" in String(e)
    expect(refused, "join after where is refused")
    refused = False
    try:
        _ = Query(left).join(right, ["tag"], ["k"])
    except e:
        refused = "different kinds" in String(e)
    expect(refused, "TEXT with INT is refused")
    refused = False
    try:
        _ = Query(left).join(right, ["nope"], ["k"])
    except e:
        refused = "no column `nope`" in String(e)
    expect(refused, "an unknown column is refused")
    print("ok test_join_api")
