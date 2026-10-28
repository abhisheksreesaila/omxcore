"""SQL text: the lexer, the parser and the syntax tree (F-150).

    var st = parse_sql("SELECT region, sum(revenue) FROM orders WHERE qty > 2 GROUP BY region ORDER BY 2 DESC LIMIT 5")

The dialect is a subset of DuckDB's: one SELECT over one named table (WHERE, GROUP BY, HAVING, ORDER BY, LIMIT,
OFFSET), optionally behind EXPLAIN. Everything else raises, naming the construct and its place in the text:

    SQL error at line 1, column 21: JOIN is not supported yet (joins arrive with F-153)

Lines and columns count from 1; a column counts characters, not bytes. The parser only builds the tree; the binder
(`bind.mojo`) gives names and types their meaning and raises for unknown columns and functions.
"""

# ── tokens ───────────────────────────────────────────────────────────────────────────────────────────

comptime T_EOF = 0
comptime T_IDENT = 1        # a word: a keyword or a name, as written (`up` is its upper case)
comptime T_QIDENT = 2       # "a quoted name"
comptime T_INT = 3          # digits
comptime T_DEC = 4          # digits with a point: 12.50, .5
comptime T_STR = 5          # 'text' (quotes removed, '' unescaped)
comptime T_SYM = 6          # ( ) , . * + - / // % = <> != < <= > >= ; :: || ? $n
comptime T_FLT = 7          # digits with an exponent: 1e5, 2.5E-3 (a DOUBLE, as DuckDB reads it)


@fieldwise_init
struct Token(Copyable, Movable):
    var kind: Int
    var text: String
    var up: String
    var line: Int
    var col: Int


def sql_error(line: Int, col: Int, message: String) -> Error:
    """Every error of the SQL front end has this shape, so a caller (the `mx` command) can read the position back."""
    return Error("SQL error at line " + String(line) + ", column " + String(col) + ": " + message)


def _tok_error(t: Token, message: String) -> Error:
    return sql_error(t.line, t.col, message)


def _is_letter(b: UInt8) -> Bool:
    return (UInt8(97) <= b <= UInt8(122)) or (UInt8(65) <= b <= UInt8(90)) or b == UInt8(95) or b >= UInt8(128)


def _is_digit(b: UInt8) -> Bool:
    return UInt8(48) <= b <= UInt8(57)


def _sub(src: Span[UInt8, _], a: Int, b: Int) -> String:
    return String(unsafe_from_utf8=src[a:b])


def upper_ascii(s: String) -> String:
    var out = String()
    for b in s.as_bytes():
        out += chr(Int(b - 32)) if UInt8(97) <= b <= UInt8(122) else chr(Int(b))
    return out^


def lower_ascii(s: String) -> String:
    var out = String()
    for b in s.as_bytes():
        out += chr(Int(b + 32)) if UInt8(65) <= b <= UInt8(90) else chr(Int(b))
    return out^


def lex_sql(text: String) raises -> List[Token]:
    """The tokens of `text`, ending with a T_EOF. Raises for an unterminated string or comment, a character SQL does not
    use, and a number written with an exponent."""
    var src = text.as_bytes()
    var n = len(src)
    var out = List[Token]()
    var i = 0
    var line = 1
    var col = 1
    while i < n:
        var b = src[i]
        var start = i
        var l0 = line
        var c0 = col
        if b == UInt8(10):
            line += 1
            col = 1
            i += 1
            continue
        if b == UInt8(32) or b == UInt8(9) or b == UInt8(13):
            i += 1
            col += 1
            continue
        if b == UInt8(45) and i + 1 < n and src[i + 1] == UInt8(45):             # -- comment
            while i < n and src[i] != UInt8(10):
                i += 1
            continue
        if b == UInt8(47) and i + 1 < n and src[i + 1] == UInt8(42):             # /* comment */
            i += 2
            col += 2
            var closed = False
            while i < n:
                if src[i] == UInt8(42) and i + 1 < n and src[i + 1] == UInt8(47):
                    i += 2
                    col += 2
                    closed = True
                    break
                if src[i] == UInt8(10):
                    line += 1
                    col = 1
                elif (src[i] & UInt8(0xC0)) != UInt8(0x80):
                    col += 1
                i += 1
            if not closed:
                raise sql_error(l0, c0, "this comment is never closed (expected */)")
            continue
        if _is_letter(b):
            while i < n and (_is_letter(src[i]) or _is_digit(src[i]) or src[i] == UInt8(36)):
                if (src[i] & UInt8(0xC0)) != UInt8(0x80):
                    col += 1
                i += 1
            var w = _sub(src, start, i)
            out.append(Token(T_IDENT, w, upper_ascii(w), l0, c0))
            continue
        if _is_digit(b) or (b == UInt8(46) and i + 1 < n and _is_digit(src[i + 1])):
            var dot = False
            while i < n and (_is_digit(src[i]) or (src[i] == UInt8(46) and not dot and i + 1 < n and _is_digit(src[i + 1]))):
                if src[i] == UInt8(46):
                    dot = True
                i += 1
                col += 1
            if i < n and (src[i] == UInt8(101) or src[i] == UInt8(69)) and i + 1 < n and \
               (_is_digit(src[i + 1]) or ((src[i + 1] == UInt8(43) or src[i + 1] == UInt8(45)) and i + 2 < n and _is_digit(src[i + 2]))):
                var j = i + 2
                while j < n and _is_digit(src[j]):
                    j += 1
                col += j - i
                i = j
                if i < n and _is_letter(src[i]):
                    raise sql_error(l0, c0, "the number `" + _sub(src, start, i) + "` runs into the letters after it")
                var fw = _sub(src, start, i)
                out.append(Token(T_FLT, fw, fw, l0, c0))
                continue
            if i < n and _is_letter(src[i]):
                raise sql_error(l0, c0, "the number `" + _sub(src, start, i) + "` runs into the letters after it")
            var w = _sub(src, start, i)
            out.append(Token(T_DEC if dot else T_INT, w, w, l0, c0))
            continue
        if b == UInt8(39) or b == UInt8(34):                                        # 'string' or "name"
            var q = b
            i += 1
            col += 1
            var s = String()
            var closed = False
            while i < n:
                if src[i] == q:
                    if i + 1 < n and src[i + 1] == q:
                        s += chr(Int(q))
                        i += 2
                        col += 2
                        continue
                    i += 1
                    col += 1
                    closed = True
                    break
                var seq = i
                i += 1
                while i < n and (src[i] & UInt8(0xC0)) == UInt8(0x80):
                    i += 1
                if src[seq] == UInt8(10):
                    line += 1
                    col = 1
                else:
                    col += 1
                s += _sub(src, seq, i)
            if not closed:
                raise sql_error(l0, c0, "this " + ("string" if q == UInt8(39) else "quoted name") + " is never closed (expected a closing " +
                                ("'" if q == UInt8(39) else "\"") + ")")
            out.append(Token(T_STR if q == UInt8(39) else T_QIDENT, s, s, l0, c0))
            continue
        # symbols
        var two = String()
        if i + 1 < n:
            two = _sub(src, i, i + 2)
        if two == "<>" or two == "!=" or two == "<=" or two == ">=" or two == "//" or two == "::" or two == "||":
            out.append(Token(T_SYM, two, two, l0, c0))
            i += 2
            col += 2
            continue
        var one = _sub(src, i, i + 1)
        if one == "(" or one == ")" or one == "," or one == "." or one == "*" or one == "+" or one == "-" or one == "/" or \
           one == "%" or one == "=" or one == "<" or one == ">" or one == ";" or one == "?":
            out.append(Token(T_SYM, one, one, l0, c0))
            i += 1
            col += 1
            continue
        if one == "$":
            raise sql_error(l0, c0, "parameters (`$1`) are not supported")
        var j = i + 1
        while j < n and (src[j] & UInt8(0xC0)) == UInt8(0x80):
            j += 1
        raise sql_error(l0, c0, "unexpected character `" + _sub(src, i, j) + "`")
    out.append(Token(T_EOF, "", "", line, col))
    return out^


comptime MAX_DEPTH = 256         # the deepest an expression may nest

# ── the syntax tree ──────────────────────────────────────────────────────────────────────────────────

comptime X_INT = 0          # text: the digits
comptime X_DEC = 1          # text: digits with a point
comptime X_STR = 2          # text: the string
comptime X_NULL = 3
comptime X_TRUE = 4
comptime X_FALSE = 5
comptime X_DATE = 6         # DATE '2024-01-31': text: the string
comptime X_COL = 7          # text: the name; text2: its qualifier ("" for none)
comptime X_NEG = 8          # a: operand
comptime X_BIN = 9          # text: + - * / // % = <> < <= > >= AND OR; a, b
comptime X_NOT = 10         # a
comptime X_ISNULL = 11      # a; flag: IS NOT NULL
comptime X_BETWEEN = 12     # a BETWEEN b AND c; flag: NOT BETWEEN
comptime X_IN = 13          # a IN (items); flag: NOT IN
comptime X_CALL = 14        # text: the function's name in upper case; items: its arguments; flag: DISTINCT; a: 1 for (*)
comptime X_FLT = 15         # text: digits with an exponent (a DOUBLE literal)
comptime X_SUBIN = 16       # a IN (SELECT ...): a: the left side; items[0]: the subquery (`st.subs`); flag: NOT IN
comptime X_EXISTS = 17      # EXISTS (SELECT ...): items[0]: the subquery; flag: NOT EXISTS
# F-156: expressions
comptime X_CASE = 18        # a: the operand of a simple CASE (-1 for a searched one); items: when, then, when, then ...; c: the ELSE (-1 for none)
comptime X_CAST = 19        # a: the operand; text: the type name in upper case (INTEGER, BIGINT, SMALLINT, TINYINT, DECIMAL, DOUBLE, REAL, VARCHAR, DATE, BOOLEAN);
                            # b: precision (-1 for none), c: scale (-1 for none); flag: TRY_CAST
comptime X_LIKE = 20        # a: the value; b: the pattern; c: the ESCAPE (-1 for none); text: LIKE or ILIKE; flag: NOT
comptime X_DISTINCT = 21    # a IS DISTINCT FROM b; flag: IS NOT DISTINCT FROM
comptime X_INTERVAL = 22    # INTERVAL n unit: a: the amount; text: the unit (DAY, WEEK, MONTH, YEAR) in upper case
comptime X_ISBOOL = 23      # a IS [NOT] TRUE / FALSE: a: the operand; text: TRUE or FALSE; flag: NOT

comptime X_SCALAR_Q = 24    # (SELECT ...) as a value (F-159): items[0]: the subquery's statement in the query's store
comptime X_IN_Q = 25        # a IN (SELECT ...): a: the left side; items[0]: the subquery's statement; flag: NOT IN
comptime X_EXISTS_Q = 26    # EXISTS (SELECT ...): items[0]: the subquery's statement; flag: NOT EXISTS

comptime JK_INNER = 0       # the kinds of join (F-154); the semi and anti joins keep only the left table's rows and columns
comptime JK_LEFT = 1
comptime JK_SEMI = 2
comptime JK_ANTI = 3
comptime JK_ANTI_NA = 4     # NOT IN (SELECT ...): an anti join that is null-aware (a NULL on either side removes the row)


@fieldwise_init
struct SqlExpr(Copyable, Movable):
    var kind: Int
    var text: String
    var text2: String
    var a: Int
    var b: Int
    var c: Int
    var items: List[Int]
    var flag: Bool
    var line: Int
    var col: Int


struct SelectItem(Copyable, Movable):
    var expr: Int
    var alias: String
    var has_alias: Bool
    var line: Int
    var col: Int
    var star: Bool       # F-152: `*` or `t.*` (expr is -1; `alias` holds the qualifier of `t.*`)

    def __init__(out self, expr: Int, alias: String, has_alias: Bool, line: Int, col: Int, star: Bool = False):
        self.expr = expr
        self.alias = alias
        self.has_alias = has_alias
        self.line = line
        self.col = col
        self.star = star


@fieldwise_init
struct OrderItem(Copyable, Movable):
    var expr: Int
    var desc: Bool
    var nulls: Int              # -1 the default (last), 0 NULLS LAST, 1 NULLS FIRST


# F-168: window frame bounds
comptime FB_UNB_PREC = 0    # UNBOUNDED PRECEDING
comptime FB_PREC = 1        # n PRECEDING
comptime FB_CUR = 2         # CURRENT ROW
comptime FB_FOL = 3         # n FOLLOWING
comptime FB_UNB_FOL = 4     # UNBOUNDED FOLLOWING
comptime FM_NONE = 0        # no frame written: the default (RANGE UNBOUNDED PRECEDING to CURRENT ROW; the whole partition with no ORDER BY)
comptime FM_ROWS = 1
comptime FM_RANGE = 2


struct WinSpec(Copyable, Movable):
    """`OVER (PARTITION BY ... ORDER BY ... [ROWS|RANGE frame])` of a call (F-168). A call refers to it by index in `SqlStatement.wins` (the call's `c`)."""
    var partition: List[Int]
    var order: List[OrderItem]
    var mode: Int
    var s_kind: Int
    var s_n: Int
    var e_kind: Int
    var e_n: Int
    var line: Int
    var col: Int

    def __init__(out self, line: Int, col: Int):
        self.partition = List[Int]()
        self.order = List[OrderItem]()
        self.mode = FM_NONE
        self.s_kind = FB_UNB_PREC
        self.s_n = 0
        self.e_kind = FB_CUR
        self.e_n = 0
        self.line = line
        self.col = col


struct TableRef(Copyable, Movable):
    """A table of FROM beyond the second (F-155): `a JOIN b ON .. JOIN c ON ..`, `a, b, c WHERE ..`."""
    var name: String
    var alias: String
    var is_file: Bool
    var line: Int
    var col: Int
    var kid: Int                # F-159: a derived table `(SELECT ...) alias`: its statement in the query's store (-1 for a named table)
    var cols: List[String]      # `alias (a, b)`: the derived table's column names

    def __init__(out self, name: String, alias: String, is_file: Bool, line: Int, col: Int, kid: Int = -1, var cols: List[String] = List[String]()):
        self.cols = cols^
        self.name = name
        self.alias = alias
        self.is_file = is_file
        self.line = line
        self.col = col
        self.kid = kid


struct SubQuery(Copyable, Movable):
    """`(SELECT item FROM table [alias] [WHERE cond])`: the only subquery shape (F-154). Its expressions live in the statement's pool."""
    var table: String
    var alias: String
    var is_file: Bool
    var line: Int
    var col: Int
    var sel: Int                # the selected expression, -1 for `*`
    var where: Int              # -1 for none
    var inner: List[Int]        # F-159: the column references of `sel` and `where` that mean the subquery's own table (a rewrite of the outer statement's names leaves them alone)
    var outer: List[Int]        # ... and the ones that mean the outer statement's (they follow its renames)

    def __init__(out self):
        self.inner = List[Int]()
        self.outer = List[Int]()
        self.table = String()
        self.alias = String()
        self.is_file = False
        self.line = 0
        self.col = 0
        self.sel = -1
        self.where = -1


comptime SET_NONE = 0           # F-159: how a statement combines two others (its `set_left` and `set_right`)
comptime SET_UNION_ALL = 1
comptime SET_UNION = 2
comptime SET_INTERSECT = 3
comptime SET_EXCEPT = 4

comptime CTE_AUTO = 0           # WITH name AS (...): inlined when read once, materialized once when read more
comptime CTE_MATERIALIZED = 1   # WITH name AS MATERIALIZED (...)
comptime CTE_INLINED = 2        # WITH name AS NOT MATERIALIZED (...)


struct Cte(Copyable, Movable):
    """`name [(columns)] AS [NOT] [MATERIALIZED] (query)` of a WITH clause (F-159)."""
    var name: String
    var columns: List[String]
    var kid: Int
    var hint: Int
    var line: Int
    var col: Int

    def __init__(out self, name: String, kid: Int, hint: Int, line: Int, col: Int):
        self.name = name
        self.columns = List[String]()
        self.kid = kid
        self.hint = hint
        self.line = line
        self.col = col


comptime EXPLAIN_NONE = 0
comptime EXPLAIN_PLAN = 1
comptime EXPLAIN_ANALYZE = 2   # F-155: the query runs and the plan shows the actual rows and milliseconds of each operator


struct SqlStatement(Movable):
    """A parsed SELECT (or EXPLAIN SELECT). Expressions live in `exprs`, referred to by index."""
    var text: String
    var exprs: List[SqlExpr]
    var items: List[SelectItem]
    var table: String
    var table_alias: String
    var table_is_file: Bool     # FROM 'orders.parquet': the name is a file path
    var table_line: Int
    var table_col: Int
    var where: Int              # -1 for none
    var group_by: List[Int]
    var having: Int
    var order_by: List[OrderItem]
    var limit: Int              # -1 for none
    var offset: Int
    var explain: Int
    var distinct: Bool          # F-152: SELECT DISTINCT
    # F-153: a second table (`a JOIN b ON ...`, `a JOIN b USING (k)`, `a, b WHERE ...`)
    var has_right: Bool
    var right_table: String
    var right_alias: String
    var right_is_file: Bool
    var right_line: Int
    var right_col: Int
    var join_on: Int            # the ON condition's expression, -1 for none
    var join_using: List[String]
    var comma_join: Bool
    var join_kind: Int          # F-154: JK_INNER, JK_LEFT, JK_SEMI, JK_ANTI
    var subs: List[SubQuery]    # F-154: the subqueries of WHERE conditions
    var extra: List[TableRef]   # F-155: the third table of FROM and the ones after it (inner joins only)
    var from_subquery: Bool     # set by the binder when it turned a WHERE subquery into the right table of a semi / anti join
    var wins: List[WinSpec]     # F-168: the OVER clauses of the calls (an X_CALL with `c` >= 0 refers to one)
    # F-159: other statements this one reads, by index in the query's store (`SqlQuery.stmts`); -1 for none
    var ctes: List[Cte]         # WITH
    var table_kid: Int          # FROM (SELECT ...) alias
    var right_kid: Int
    var set_op: Int             # SET_*: this statement is `set_left <op> set_right` (then it has no FROM; its ORDER BY and LIMIT apply to the result)
    var set_left: Int
    var set_right: Int
    var qualify: Int            # QUALIFY condition, -1 for none
    var table_cols: List[String]    # `(SELECT ...) alias (a, b)`: column names of the derived table in FROM (the right table's: `right_cols`)
    var right_cols: List[String]

    def __init__(out self, text: String):
        self.table_cols = List[String]()
        self.right_cols = List[String]()
        self.ctes = List[Cte]()
        self.table_kid = -1
        self.right_kid = -1
        self.set_op = SET_NONE
        self.set_left = -1
        self.set_right = -1
        self.qualify = -1
        self.wins = List[WinSpec]()
        self.distinct = False
        self.has_right = False
        self.right_table = String()
        self.right_alias = String()
        self.right_is_file = False
        self.right_line = 0
        self.right_col = 0
        self.join_on = -1
        self.join_using = List[String]()
        self.comma_join = False
        self.join_kind = JK_INNER
        self.subs = List[SubQuery]()
        self.extra = List[TableRef]()
        self.from_subquery = False
        self.text = text
        self.exprs = List[SqlExpr]()
        self.items = List[SelectItem]()
        self.table = String()
        self.table_alias = String()
        self.table_is_file = False
        self.table_line = 0
        self.table_col = 0
        self.where = -1
        self.group_by = List[Int]()
        self.having = -1
        self.order_by = List[OrderItem]()
        self.limit = -1
        self.offset = 0
        self.explain = EXPLAIN_NONE

    def render(self, e: Int) -> String:
        """The expression as SQL text (names as written, operators spaced), for column names and EXPLAIN."""
        ref x = self.exprs[e]
        if x.kind == X_INT or x.kind == X_DEC or x.kind == X_FLT:
            return x.text
        if x.kind == X_STR:
            return "'" + x.text.replace("'", "''") + "'"
        if x.kind == X_NULL:
            return "NULL"
        if x.kind == X_TRUE:
            return "true"
        if x.kind == X_FALSE:
            return "false"
        if x.kind == X_DATE:
            return "DATE '" + x.text + "'"
        if x.kind == X_COL:
            return (x.text2 + "." if x.text2.byte_length() > 0 else "") + x.text
        if x.kind == X_NEG:
            return "-" + self.render_operand(x.a)
        if x.kind == X_BIN:
            return self.render_operand(x.a) + " " + x.text + " " + self.render_operand(x.b)
        if x.kind == X_NOT:
            return "NOT " + self.render_operand(x.a)
        if x.kind == X_ISNULL:
            return self.render_operand(x.a) + (" IS NOT NULL" if x.flag else " IS NULL")
        if x.kind == X_BETWEEN:
            return self.render_operand(x.a) + (" NOT BETWEEN " if x.flag else " BETWEEN ") + self.render_operand(x.b) + " AND " + self.render_operand(x.c)
        if x.kind == X_IN:
            var s = self.render_operand(x.a) + (" NOT IN (" if x.flag else " IN (")
            for k in range(len(x.items)):
                s += (", " if k > 0 else "") + self.render(x.items[k])
            return s + ")"
        if x.kind == X_SUBIN:
            return self.render_operand(x.a) + (" NOT IN (SELECT ...)" if x.flag else " IN (SELECT ...)")
        if x.kind == X_EXISTS or x.kind == X_EXISTS_Q:
            return "NOT EXISTS (SELECT ...)" if x.flag else "EXISTS (SELECT ...)"
        if x.kind == X_SCALAR_Q:
            return "(SELECT ...)"
        if x.kind == X_IN_Q:
            return self.render_operand(x.a) + (" NOT IN (SELECT ...)" if x.flag else " IN (SELECT ...)")
        if x.kind == X_CASE:
            var s = String("CASE")
            if x.a >= 0:
                s += " " + self.render_operand(x.a)
            var k = 0
            while k + 1 < len(x.items):
                s += " WHEN " + self.render(x.items[k]) + " THEN " + self.render(x.items[k + 1])
                k += 2
            if x.c >= 0:
                s += " ELSE " + self.render(x.c)
            return s + " END"
        if x.kind == X_CAST:
            var ty = x.text
            if x.b >= 0:
                ty += "(" + String(x.b) + ("," + String(x.c) if x.c >= 0 else "") + ")"
            return ("TRY_CAST(" if x.flag else "CAST(") + self.render(x.a) + " AS " + ty + ")"
        if x.kind == X_LIKE:
            return self.render_operand(x.a) + (" NOT " if x.flag else " ") + x.text + " " + self.render_operand(x.b) + ((" ESCAPE " + self.render(x.c)) if x.c >= 0 else "")
        if x.kind == X_DISTINCT:
            return self.render_operand(x.a) + (" IS NOT DISTINCT FROM " if x.flag else " IS DISTINCT FROM ") + self.render_operand(x.b)
        if x.kind == X_INTERVAL:
            return "INTERVAL " + self.render_operand(x.a) + " " + x.text
        if x.kind == X_ISBOOL:
            return self.render_operand(x.a) + (" IS NOT " if x.flag else " IS ") + x.text
        var s = x.text.lower() + "("
        if x.a == 1:
            s += "*"
        elif x.flag:
            s += "DISTINCT "
        for k in range(len(x.items)):
            s += (", " if k > 0 else "") + self.render(x.items[k])
        s += ")"
        if x.c >= 0:
            s += " OVER (" + self.render_window(x.c) + ")"
        return s

    def render_window(self, w: Int) -> String:
        """The inside of an OVER clause as SQL text (the name DuckDB gives a window column)."""
        ref ws = self.wins[w]
        var s = String()
        if len(ws.partition) > 0:
            s += "PARTITION BY "
            for k in range(len(ws.partition)):
                s += (", " if k > 0 else "") + self.render(ws.partition[k])
        if len(ws.order) > 0:
            s += (" " if s.byte_length() > 0 else "") + "ORDER BY "
            for k in range(len(ws.order)):
                s += (", " if k > 0 else "") + self.render(ws.order[k].expr) + (" DESC" if ws.order[k].desc else "")
                if ws.order[k].nulls == 1:
                    s += " NULLS FIRST"
                elif ws.order[k].nulls == 0:
                    s += " NULLS LAST"
        if ws.mode != FM_NONE:
            s += (" " if s.byte_length() > 0 else "") + ("ROWS" if ws.mode == FM_ROWS else "RANGE") + " BETWEEN " + _bound_text(ws.s_kind, ws.s_n) + " AND " + _bound_text(ws.e_kind, ws.e_n)
        return s

    def render_operand(self, e: Int) -> String:
        var k = self.exprs[e].kind
        if k == X_BIN or k == X_NOT or k == X_ISNULL or k == X_BETWEEN or k == X_IN or k == X_NEG or k == X_SUBIN or k == X_IN_Q or k == X_LIKE or k == X_DISTINCT or k == X_ISBOOL:
            return "(" + self.render(e) + ")"
        return self.render(e)


struct SqlQuery(Movable):
    """A parsed query (F-159): every statement it is made of in one store, the one the text ends with last. A derived table, a CTE, a subquery or an
    operand of UNION is a statement of its own (its own expressions); the statements that read it hold its index."""
    var stmts: List[SqlStatement]

    def __init__(out self):
        self.stmts = List[SqlStatement]()

    def root(self) -> Int:
        return len(self.stmts) - 1


def _bound_text(kind: Int, n: Int) -> String:
    if kind == FB_UNB_PREC:
        return "UNBOUNDED PRECEDING"
    if kind == FB_UNB_FOL:
        return "UNBOUNDED FOLLOWING"
    if kind == FB_CUR:
        return "CURRENT ROW"
    return String(n) + (" PRECEDING" if kind == FB_PREC else " FOLLOWING")


# ── the parser ───────────────────────────────────────────────────────────────────────────────────────

def _reserved(w: String) -> Bool:
    """Words that end a select item or a table name, so they are never taken for an alias."""
    return (w == "FROM" or w == "WHERE" or w == "GROUP" or w == "HAVING" or w == "ORDER" or w == "LIMIT" or w == "OFFSET" or
            w == "UNION" or w == "INTERSECT" or w == "EXCEPT" or w == "WINDOW" or w == "QUALIFY" or w == "JOIN" or
            w == "INNER" or w == "LEFT" or w == "RIGHT" or w == "FULL" or w == "CROSS" or w == "NATURAL" or w == "ON" or
            w == "USING" or w == "AS" or w == "AND" or w == "OR" or w == "NOT" or w == "IN" or w == "IS" or w == "BETWEEN" or
            w == "LIKE" or w == "ILIKE" or w == "SELECT" or w == "ASC" or w == "DESC" or w == "NULLS" or w == "FETCH" or
            w == "TABLESAMPLE" or w == "SAMPLE" or w == "FILTER" or w == "OVER" or w == "WITHIN" or w == "THEN" or
            w == "ELSE" or w == "END" or w == "WHEN" or w == "ESCAPE" or w == "SIMILAR" or w == "GLOB" or w == "ANTI" or
            w == "SEMI" or w == "ASOF" or w == "POSITIONAL" or w == "LATERAL" or w == "OUTER")


def edit_distance(a: String, b: String) -> Int:
    """Levenshtein distance of two strings (bytes), for the "did you mean" hints."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    var prev = List[Int](length=len(y) + 1, fill=0)
    var cur = List[Int](length=len(y) + 1, fill=0)
    for j in range(len(y) + 1):
        prev[j] = j
    for i in range(len(x)):
        cur[0] = i + 1
        for j in range(len(y)):
            var cost = 0 if x[i] == y[j] else 1
            cur[j + 1] = min(min(prev[j + 1] + 1, cur[j] + 1), prev[j] + cost)
        swap(prev, cur)
    return prev[len(y)]


def closest(word: String, candidates: List[String]) -> String:
    """The candidate nearest to `word` (case ignored) when it is near enough to be a typo, else ""."""
    var best = String()
    var best_d = 1000
    var w = lower_ascii(word)
    for c in candidates:
        var d = edit_distance(w, lower_ascii(c))
        if d < best_d:
            best_d = d
            best = c
    var limit = 1 if w.byte_length() <= 3 else 2 if w.byte_length() <= 7 else 3
    return best if best_d <= limit and best_d < w.byte_length() else String()


struct _Parser(Movable):
    var toks: List[Token]
    var pos: Int
    var st: SqlStatement
    var depths: List[Int]            # the depth of each expression of `st.exprs`
    var level: Int                   # nesting of the recursive descent now
    var store: List[SqlStatement]    # F-159: the finished statements of a query (see SqlQuery); a nested query is parsed by a parser of its own that borrows it
    var ref_kid: Int                 # the statement of the derived table `table_ref` just read (-1 for a named table)
    var ref_cols: List[String]       # ... and the column names after its alias

    def __init__(out self, var toks: List[Token], text: String):
        self.toks = toks^
        self.pos = 0
        self.depths = List[Int]()
        self.level = 0
        self.st = SqlStatement(text)
        self.store = List[SqlStatement]()
        self.ref_kid = -1
        self.ref_cols = List[String]()

    def take(deinit self) -> SqlStatement:
        return self.st^

    # -- cursor
    def tok(self) -> Token:
        return self.toks[self.pos].copy()

    def peek_up(self, ahead: Int = 0) -> String:
        var at = min(self.pos + ahead, len(self.toks) - 1)
        return self.toks[at].up if self.toks[at].kind == T_IDENT else String()

    def is_sym(self, s: String, ahead: Int = 0) -> Bool:
        var at = min(self.pos + ahead, len(self.toks) - 1)
        return self.toks[at].kind == T_SYM and self.toks[at].text == s

    def is_kw(self, w: String, ahead: Int = 0) -> Bool:
        return self.peek_up(ahead) == w

    def advance(mut self) -> Token:
        var t = self.toks[self.pos].copy()
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t^

    def describe(self, t: Token) -> String:
        if t.kind == T_EOF:
            return "the end of the statement"
        if t.kind == T_STR:
            return "the string '" + t.text + "'"
        if t.kind == T_QIDENT:
            return "the name \"" + t.text + "\""
        return "`" + t.text + "`"

    def expected(self, what: String) -> Error:
        var t = self.tok()
        var msg = "expected " + what + ", found " + self.describe(t)
        if t.kind == T_IDENT and what.byte_length() > 0 and UInt8(65) <= what.as_bytes()[0] <= UInt8(90):
            var hints: List[String] = ["SELECT", "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET", "AND", "OR", "AS", "BY"]
            var near = closest(t.text, hints)
            if near.byte_length() > 0:
                msg += " (did you mean " + near + "?)"
        return _tok_error(t, msg)

    def unsupported(self, t: Token, what: String, when: String) -> Error:
        return _tok_error(t, what + " is not supported yet" + ((" (" + when + ")") if when.byte_length() > 0 else ""))

    def expect_kw(mut self, w: String) raises -> Token:
        if not self.is_kw(w):
            raise self.expected(w)
        return self.advance()

    def expect_sym(mut self, s: String) raises -> Token:
        if not self.is_sym(s):
            raise self.expected("`" + s + "`")
        return self.advance()

    def enter(mut self, t: Token) raises:
        """Counts one more level of nesting; the parser, the binder and the emitter recurse, so the depth is bounded."""
        self.level += 1
        if self.level > MAX_DEPTH:
            raise _tok_error(t, "this expression is nested more than " + String(MAX_DEPTH) + " levels deep (parentheses, NOT or a minus sign repeated)")

    def add(mut self, var x: SqlExpr) -> Int:
        self.st.exprs.append(x^)
        return len(self.st.exprs) - 1

    def node(mut self, kind: Int, text: String, a: Int, b: Int, c: Int, t: Token, flag: Bool = False) raises -> Int:
        var d = 0
        for k in [a, b, c]:
            if k >= 0 and kind != X_CALL:                   # (a call's `a` is a flag, its arguments are items)
                d = max(d, self.depths[k])
        if d + 1 > MAX_DEPTH:
            raise _tok_error(t, "this expression is " + String(d + 1) + " levels deep, more than the " + String(MAX_DEPTH) +
                             " allowed (a chain of that many terms joined by `" + (text if text.byte_length() > 0 else "an operator") + "` counts)")
        self.depths.append(d + 1)
        return self.add(SqlExpr(kind, text, String(), a, b, c, List[Int](), flag, t.line, t.col))

    def node_items(mut self, kind: Int, text: String, a: Int, b: Int, c: Int, var items: List[Int], t: Token, flag: Bool = False) raises -> Int:
        """A node whose operands are a list (CASE): its depth counts its items too."""
        var d = 0
        for k in [a, b, c]:
            if k >= 0:
                d = max(d, self.depths[k])
        for k in items:
            d = max(d, self.depths[k])
        if d + 1 > MAX_DEPTH:
            raise _tok_error(t, "this expression is " + String(d + 1) + " levels deep, more than the " + String(MAX_DEPTH) + " allowed")
        self.depths.append(d + 1)
        return self.add(SqlExpr(kind, text, String(), a, b, c, items^, flag, t.line, t.col))

    # -- statement
    def statement(mut self) raises:
        var t = self.tok()
        if t.kind == T_IDENT and t.up == "EXPLAIN":
            _ = self.advance()
            self.st.explain = EXPLAIN_PLAN
            if self.is_kw("ANALYZE"):
                _ = self.advance()
                self.st.explain = EXPLAIN_ANALYZE
            t = self.tok()
        if (t.kind == T_IDENT and (t.up == "SELECT" or t.up == "WITH")) or (t.kind == T_SYM and t.text == "("):
            self.query_expr()
        elif t.kind == T_IDENT and (t.up == "INSERT" or t.up == "UPDATE" or t.up == "DELETE" or t.up == "CREATE" or t.up == "DROP" or
                                    t.up == "ALTER" or t.up == "COPY" or t.up == "VALUES" or t.up == "SET" or t.up == "PRAGMA" or
                                    t.up == "ATTACH" or t.up == "DESCRIBE" or t.up == "SHOW" or t.up == "PIVOT" or t.up == "FROM" or
                                    t.up == "TABLE"):
            raise _tok_error(t, "`" + t.up + "` statements are not supported: only SELECT (and EXPLAIN SELECT) is")
        else:
            raise self.expected("SELECT" if self.st.explain != EXPLAIN_NONE else "SELECT or EXPLAIN")
        if self.is_sym(";"):
            _ = self.advance()
        if self.tok().kind != T_EOF:
            raise self.expected("the end of the statement")

    def has_set_op(self) -> Bool:
        """Whether UNION, INTERSECT or EXCEPT comes before this query's end (outside parentheses)."""
        var depth = 0
        var i = self.pos
        while i < len(self.toks):
            var t = self.toks[i].copy()
            if t.kind == T_EOF or (t.kind == T_SYM and t.text == ";"):
                return False
            if t.kind == T_SYM and t.text == "(":
                depth += 1
            elif t.kind == T_SYM and t.text == ")":
                depth -= 1
                if depth < 0:
                    return False
            elif depth == 0 and t.kind == T_IDENT and (t.up == "UNION" or t.up == "INTERSECT" or t.up == "EXCEPT"):
                return True
            i += 1
        return False

    def nested(mut self) -> _Parser:
        """A parser for a query inside this one, at the same place and sharing the store (give the store back with `finish_nested`)."""
        var sub = _Parser(self.toks.copy(), self.st.text)
        sub.pos = self.pos
        sub.level = self.level + 1
        swap(sub.store, self.store)
        return sub^

    def finish_nested(mut self, mut sub: _Parser) -> Int:
        """Takes the store and the position back from a nested parser and stores its statement; returns the statement's index."""
        swap(sub.store, self.store)
        self.pos = sub.pos
        var done = SqlStatement(self.st.text)
        swap(done, sub.st)
        self.store.append(done^)
        return len(self.store) - 1

    def query_after_paren(mut self) raises -> Int:
        """After `(`: a whole query up to and including the `)`; returns its statement's index."""
        self.enter(self.tok())
        var sub = self.nested()
        try:
            sub.query_expr()
        except e:
            swap(sub.store, self.store)
            raise e^
        var k = self.finish_nested(sub)
        _ = self.expect_sym(")")
        self.level -= 1
        return k

    def set_operand(mut self) raises -> Int:
        if self.is_sym("("):
            _ = self.advance()
            if not (self.is_kw("SELECT") or self.is_kw("WITH") or self.is_sym("(")):
                raise self.expected("SELECT")
            return self.query_after_paren()
        if not self.is_kw("SELECT"):
            raise self.expected("SELECT")
        var sub = self.nested()
        try:
            sub.select_core()
        except e:
            swap(sub.store, self.store)
            raise e^
        return self.finish_nested(sub)

    def store_setop(mut self, op: Int, left: Int, right: Int, t: Token) -> Int:
        var n = SqlStatement(self.st.text)
        n.set_op = op
        n.set_left = left
        n.set_right = right
        n.table_line = t.line
        n.table_col = t.col
        self.store.append(n^)
        return len(self.store) - 1

    def set_kind(mut self, word: String, t: Token) raises -> Int:
        """After UNION / INTERSECT / EXCEPT: ALL or DISTINCT; the operation."""
        var all = False
        if self.is_kw("ALL"):
            _ = self.advance()
            all = True
        elif self.is_kw("DISTINCT"):
            _ = self.advance()
        if self.is_kw("BY"):
            raise self.unsupported(self.tok(), "`" + word + " BY NAME`", "")
        if word == "UNION":
            return SET_UNION_ALL if all else SET_UNION
        if all:
            raise self.unsupported(t, "`" + word + " ALL`", "")
        return SET_INTERSECT if word == "INTERSECT" else SET_EXCEPT

    def intersect_chain(mut self) raises -> Int:
        var left = self.set_operand()
        while self.is_kw("INTERSECT"):
            var t = self.advance()
            var op = self.set_kind("INTERSECT", t)
            var right = self.set_operand()
            left = self.store_setop(op, left, right, t)
        return left

    def set_chain(mut self) raises:
        """`a UNION [ALL] b INTERSECT c EXCEPT d ...` (INTERSECT binds tighter) into this statement, left to right."""
        var left = self.intersect_chain()
        var have = False
        var pop = SET_NONE
        var pl = -1
        var pr = -1
        var pline = self.tok().line
        var pcol = self.tok().col
        while self.is_kw("UNION") or self.is_kw("EXCEPT"):
            if have:
                left = self.store_setop(pop, pl, pr, Token(T_EOF, String(), String(), pline, pcol))
            var t = self.advance()
            var op = self.set_kind(t.up, t)
            var right = self.intersect_chain()
            have = True
            pop = op
            pl = left
            pr = right
            pline = t.line
            pcol = t.col
        if not have:                                                            # an INTERSECT chain: its last node is this statement
            var n = self.store.pop()
            pop = n.set_op
            pl = n.set_left
            pr = n.set_right
            pline = n.table_line
            pcol = n.table_col
        self.st.table_line = pline
        self.st.table_col = pcol
        self.st.set_op = pop
        self.st.set_left = pl
        self.st.set_right = pr

    def with_clause(mut self) raises:
        _ = self.advance()                                                      # WITH
        if self.is_kw("RECURSIVE"):
            raise self.unsupported(self.tok(), "`WITH RECURSIVE`", "")
        while True:
            var nm = self.tok()
            if nm.kind != T_IDENT and nm.kind != T_QIDENT:
                raise self.expected("a name for the common table expression")
            if nm.kind == T_IDENT and _reserved(nm.up):
                raise self.expected("a name for the common table expression")
            _ = self.advance()
            var cols = List[String]()
            if self.is_sym("("):
                _ = self.advance()
                while True:
                    var c = self.tok()
                    if c.kind != T_IDENT and c.kind != T_QIDENT:
                        raise self.expected("a column name")
                    cols.append(c.text)
                    _ = self.advance()
                    if self.is_sym(","):
                        _ = self.advance()
                        continue
                    break
                _ = self.expect_sym(")")
            _ = self.expect_kw("AS")
            var hint = CTE_AUTO
            if self.is_kw("NOT") and self.is_kw("MATERIALIZED", 1):
                _ = self.advance()
                _ = self.advance()
                hint = CTE_INLINED
            elif self.is_kw("MATERIALIZED"):
                _ = self.advance()
                hint = CTE_MATERIALIZED
            _ = self.expect_sym("(")
            if not (self.is_kw("SELECT") or self.is_kw("WITH") or self.is_sym("(")):
                raise self.expected("SELECT")
            var k = self.query_after_paren()
            var c = Cte(nm.text, k, hint, nm.line, nm.col)
            c.columns = cols^
            self.st.ctes.append(c^)
            if self.is_sym(","):
                _ = self.advance()
                continue
            break

    def query_expr(mut self) raises:
        """`[WITH ...] select | (query) [UNION | INTERSECT | EXCEPT ...] [ORDER BY ...] [LIMIT ...]` into this statement."""
        if self.is_kw("WITH"):
            self.with_clause()
        if self.has_set_op():
            self.set_chain()
        elif self.is_sym("("):                                                  # (query) ORDER BY ..: a derived table read whole
            var at = self.advance()
            if not (self.is_kw("SELECT") or self.is_kw("WITH") or self.is_sym("(")):
                raise self.expected("SELECT")
            var k = self.query_after_paren()
            self.st.items.append(SelectItem(-1, String(), False, at.line, at.col, True))
            self.st.table_kid = k
            self.st.table_line = at.line
            self.st.table_col = at.col
        elif self.is_kw("SELECT"):
            self.select_core()
        else:
            raise self.expected("SELECT")
        self.order_limit()

    def select_core(mut self) raises:
        _ = self.advance()                                                      # SELECT
        if self.is_kw("DISTINCT"):
            if self.is_kw("ON", 1):
                raise self.unsupported(self.tok(), "`SELECT DISTINCT ON`", "")
            _ = self.advance()
            self.st.distinct = True
        elif self.is_kw("ALL"):
            _ = self.advance()
        while True:
            var t = self.tok()
            if self.is_sym("*") or ((t.kind == T_IDENT or t.kind == T_QIDENT) and self.is_sym(".", 1) and self.is_sym("*", 2)):
                var qual = String()
                if not self.is_sym("*"):
                    qual = t.text
                    _ = self.advance()
                    _ = self.advance()
                _ = self.advance()
                self.st.items.append(SelectItem(-1, qual, False, t.line, t.col, True))
                if self.is_sym(","):
                    _ = self.advance()
                    continue
                break
            var e = self.expr()
            var alias = String()
            var has = False
            if self.is_kw("AS"):
                _ = self.advance()
                var a = self.tok()
                if a.kind != T_IDENT and a.kind != T_QIDENT and a.kind != T_STR:
                    raise self.expected("a name after AS")
                alias = a.text
                has = True
                _ = self.advance()
            elif (self.tok().kind == T_IDENT and not _reserved(self.tok().up)) or self.tok().kind == T_QIDENT:
                self.reject_misspelled_clause()
                alias = self.tok().text
                has = True
                _ = self.advance()
            self.st.items.append(SelectItem(e, alias, has, t.line, t.col))
            if self.is_sym(","):
                _ = self.advance()
                continue
            break
        if not self.is_kw("FROM"):
            if self.tok().kind == T_EOF or self.is_sym(";"):
                raise _tok_error(self.tok(), "SELECT without FROM is not supported yet (F-152)")
            raise self.expected("FROM")
        _ = self.advance()
        var tr = self.table_ref()
        self.st.table = tr[0]
        self.st.table_is_file = tr[2]
        self.st.table_line = tr[3]
        self.st.table_col = tr[4]
        self.st.table_alias = tr[1]
        self.st.table_kid = self.ref_kid
        self.st.table_cols = self.ref_cols.copy()
        self.join_tail()
        if self.is_kw("WHERE"):
            _ = self.advance()
            self.st.where = self.expr()
        if self.is_kw("GROUP"):
            _ = self.advance()
            _ = self.expect_kw("BY")
            if self.is_kw("ALL"):
                raise self.unsupported(self.tok(), "`GROUP BY ALL`", "")
            if self.is_kw("ROLLUP") or self.is_kw("CUBE") or self.is_kw("GROUPING"):
                raise self.unsupported(self.tok(), "`GROUP BY " + self.tok().up + "`", "")
            while True:
                self.st.group_by.append(self.expr())
                if self.is_sym(","):
                    _ = self.advance()
                    continue
                break
        if self.is_kw("HAVING"):
            _ = self.advance()
            self.st.having = self.expr()
        if self.is_kw("QUALIFY"):
            _ = self.advance()
            self.st.qualify = self.expr()
        if self.is_kw("WINDOW") or self.is_kw("TABLESAMPLE") or self.is_kw("SAMPLE"):
            raise self.unsupported(self.tok(), "`" + self.tok().up + "`", "")

    def order_limit(mut self) raises:
        if self.is_kw("ORDER"):
            _ = self.advance()
            _ = self.expect_kw("BY")
            if self.is_kw("ALL"):
                raise self.unsupported(self.tok(), "`ORDER BY ALL`", "")
            while True:
                self.st.order_by.append(self.order_item())
                if self.is_sym(","):
                    _ = self.advance()
                    continue
                break
        # LIMIT and OFFSET in either order
        for _ in range(2):
            if self.is_kw("LIMIT") and self.st.limit < 0:
                _ = self.advance()
                self.st.limit = self.count_literal("LIMIT")
            elif self.is_kw("OFFSET") and self.st.offset == 0:
                _ = self.advance()
                self.st.offset = self.count_literal("OFFSET")
        if self.is_kw("FETCH"):
            raise self.unsupported(self.tok(), "`FETCH`", "use LIMIT")

    def order_item(mut self) raises -> OrderItem:
        """`expr [ASC | DESC] [NULLS FIRST | LAST]`."""
        var e = self.expr()
        var desc = False
        var nulls = -1
        if self.is_kw("ASC"):
            _ = self.advance()
        elif self.is_kw("DESC"):
            _ = self.advance()
            desc = True
        if self.is_kw("NULLS"):
            _ = self.advance()
            if self.is_kw("FIRST"):
                nulls = 1
            elif self.is_kw("LAST"):
                nulls = 0
            else:
                raise self.expected("FIRST or LAST after NULLS")
            _ = self.advance()
        return OrderItem(e, desc, nulls)

    def frame_bound(mut self, mode: Int, start: Bool, mut kind: Int, mut n: Int) raises:
        var t = self.tok()
        if self.is_kw("UNBOUNDED"):
            _ = self.advance()
            if self.is_kw("PRECEDING"):
                kind = FB_UNB_PREC
            elif self.is_kw("FOLLOWING"):
                kind = FB_UNB_FOL
            else:
                raise self.expected("PRECEDING or FOLLOWING after UNBOUNDED")
            _ = self.advance()
        elif self.is_kw("CURRENT"):
            _ = self.advance()
            _ = self.expect_kw("ROW")
            kind = FB_CUR
        elif t.kind == T_INT:
            _ = self.advance()
            if mode == FM_RANGE:
                raise self.unsupported(t, "`RANGE` with an offset (" + t.text + ")", "use ROWS, or UNBOUNDED / CURRENT ROW bounds")
            if t.text.byte_length() > 15:
                raise _tok_error(t, "the frame offset " + t.text + " is too large")
            n = Int(t.text)
            if self.is_kw("PRECEDING"):
                kind = FB_PREC
            elif self.is_kw("FOLLOWING"):
                kind = FB_FOL
            else:
                raise self.expected("PRECEDING or FOLLOWING after the offset")
            _ = self.advance()
        else:
            raise self.unsupported(t, "a frame offset that is not a whole-number literal (" + self.describe(t) + ")", "write ROWS BETWEEN 2 PRECEDING AND CURRENT ROW")
        if start and kind == FB_UNB_FOL:
            raise _tok_error(t, "a window frame cannot start at UNBOUNDED FOLLOWING")
        if (not start) and kind == FB_UNB_PREC:
            raise _tok_error(t, "a window frame cannot end at UNBOUNDED PRECEDING")

    def window_spec(mut self) raises -> Int:
        """After OVER: `(PARTITION BY a, b ORDER BY c [ROWS | RANGE [BETWEEN x AND y | x]])` (F-168). Named windows, GROUPS frames and EXCLUDE raise by name."""
        var t = self.tok()
        if not self.is_sym("("):
            if t.kind == T_IDENT:
                raise self.unsupported(t, "`OVER " + t.text + "` (a named window)", "write the window out: OVER (PARTITION BY ... ORDER BY ...)")
            raise self.expected("`(` after OVER")
        _ = self.advance()
        var w = WinSpec(t.line, t.col)
        if self.is_kw("PARTITION"):
            _ = self.advance()
            _ = self.expect_kw("BY")
            while True:
                w.partition.append(self.expr())
                if self.is_sym(","):
                    _ = self.advance()
                    continue
                break
        if self.is_kw("ORDER"):
            _ = self.advance()
            _ = self.expect_kw("BY")
            while True:
                w.order.append(self.order_item())
                if self.is_sym(","):
                    _ = self.advance()
                    continue
                break
        if self.is_kw("GROUPS"):
            raise self.unsupported(self.tok(), "a `GROUPS` frame", "use ROWS or RANGE")
        if self.is_kw("ROWS") or self.is_kw("RANGE"):
            w.mode = FM_ROWS if self.is_kw("ROWS") else FM_RANGE
            _ = self.advance()
            if self.is_kw("BETWEEN"):
                _ = self.advance()
                self.frame_bound(w.mode, True, w.s_kind, w.s_n)
                _ = self.expect_kw("AND")
                self.frame_bound(w.mode, False, w.e_kind, w.e_n)
            else:
                self.frame_bound(w.mode, True, w.s_kind, w.s_n)
                w.e_kind = FB_CUR
                if w.s_kind == FB_UNB_FOL:
                    raise _tok_error(t, "a window frame cannot start at UNBOUNDED FOLLOWING")
        if w.mode != FM_NONE:                                                   # (the combinations DuckDB's parser refuses, with its words)
            if w.s_kind == FB_FOL and w.e_kind == FB_CUR:
                raise _tok_error(t, "frame starting from following row cannot end with current row")
            if w.s_kind == FB_FOL and w.e_kind == FB_PREC:
                raise _tok_error(t, "frame starting from following row cannot have preceding rows")
            if w.s_kind == FB_CUR and w.e_kind == FB_PREC:
                raise _tok_error(t, "frame starting from current row cannot have preceding rows")
        if self.is_kw("EXCLUDE"):
            raise self.unsupported(self.tok(), "`EXCLUDE` in a window frame", "")
        _ = self.expect_sym(")")
        self.st.wins.append(w^)
        return len(self.st.wins) - 1

    def reject_misspelled_clause(self) raises:
        """A word that is not a keyword but nearly is (`form`, `wher`), right before a name or a number, is a typo, not an alias."""
        var t = self.tok()
        if t.kind != T_IDENT:
            return
        var nxt = self.toks[min(self.pos + 1, len(self.toks) - 1)].copy()
        if nxt.kind == T_EOF or (nxt.kind == T_SYM and nxt.text != "(") or (nxt.kind == T_IDENT and _reserved(nxt.up)):
            return
        var clauses: List[String] = ["FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET"]
        var near = closest(t.text, clauses)
        if near.byte_length() > 0:
            raise _tok_error(t, "expected " + ("FROM" if self.st.table.byte_length() == 0 else "a keyword or the end of the statement") +
                             ", found `" + t.text + "` (did you mean " + near + "?)")

    def count_literal(mut self, what: String) raises -> Int:
        var t = self.tok()
        if t.kind != T_INT:
            if t.kind == T_IDENT and t.up == "ALL" and what == "LIMIT":
                raise self.unsupported(t, "`LIMIT ALL`", "")
            raise _tok_error(t, what + " needs a whole number, found " + self.describe(t) + " (expressions are not supported there)")
        _ = self.advance()
        if t.text.byte_length() > 18:
            raise _tok_error(t, what + " " + t.text + " is too large")
        var v = 0
        for b in t.text.as_bytes():
            v = v * 10 + Int(b - 48)
        if self.is_sym("%") or self.is_kw("PERCENT"):
            raise self.unsupported(self.tok(), "`" + what + "` as a percentage", "")
        return v

    def table_ref(mut self) raises -> Tuple[String, String, Bool, Int, Int]:
        """One table in FROM: (name, alias, is a file name, line, column)."""
        var tt = self.tok()
        self.ref_kid = -1
        self.ref_cols = List[String]()
        if self.is_sym("("):
            if not (self.is_kw("SELECT", 1) or self.is_kw("WITH", 1) or self.is_sym("(", 1)):
                raise self.unsupported(tt, "a parenthesized table or join", "")
            _ = self.advance()
            var k = self.query_after_paren()
            var dalias = String()
            if self.is_kw("AS"):
                _ = self.advance()
                var a = self.tok()
                if a.kind != T_IDENT and a.kind != T_QIDENT:
                    raise self.expected("a name after AS")
                dalias = a.text
                _ = self.advance()
            elif (self.tok().kind == T_IDENT and not _reserved(self.tok().up)) or self.tok().kind == T_QIDENT:
                dalias = self.tok().text
                _ = self.advance()
            if self.is_sym("("):
                if dalias.byte_length() == 0:
                    raise self.expected("a table alias before the column names")
                _ = self.advance()
                while True:
                    var cn = self.tok()
                    if cn.kind != T_IDENT and cn.kind != T_QIDENT:
                        raise self.expected("a column name")
                    self.ref_cols.append(cn.text)
                    _ = self.advance()
                    if self.is_sym(","):
                        _ = self.advance()
                        continue
                    break
                _ = self.expect_sym(")")
            self.ref_kid = k
            return (String(), dalias, False, tt.line, tt.col)
        if tt.kind == T_IDENT and tt.up == "LATERAL":
            raise self.unsupported(tt, "`LATERAL`", "")
        if tt.kind != T_IDENT and tt.kind != T_QIDENT and tt.kind != T_STR:
            raise self.expected("a table name or 'file name'")
        if tt.kind == T_IDENT and _reserved(tt.up):
            raise self.expected("a table name")
        _ = self.advance()
        if self.is_sym("."):
            raise self.unsupported(self.tok(), "a schema-qualified table name", "")
        var alias = String()
        if self.is_kw("AS"):
            _ = self.advance()
            var a = self.tok()
            if a.kind != T_IDENT and a.kind != T_QIDENT:
                raise self.expected("a name after AS")
            alias = a.text
            _ = self.advance()
        elif (self.tok().kind == T_IDENT and not _reserved(self.tok().up)) or self.tok().kind == T_QIDENT:
            self.reject_misspelled_clause()
            alias = self.tok().text
            _ = self.advance()
        return (tt.text, alias, tt.kind == T_STR, tt.line, tt.col)

    def join_tail(mut self) raises:
        """After the first table: `, b`, `[INNER] JOIN b ON cond`, `[INNER] JOIN b USING (k, ...)`, `LEFT [OUTER] / SEMI / ANTI JOIN b ON ..`, and (F-155) more of
        the inner kinds after them: `a JOIN b ON .. JOIN c ON ..`, `a, b, c WHERE ..`. Every other join raises by name."""
        while True:
            var t = self.tok()
            var inner_kw = t.kind == T_IDENT and (t.up == "JOIN" or (t.up == "INNER" and self.is_kw("JOIN", 1)))
            var outer_kw = t.kind == T_IDENT and ((t.up == "LEFT" or t.up == "SEMI" or t.up == "ANTI") and
                                                  (self.is_kw("JOIN", 1) or (t.up == "LEFT" and self.is_kw("OUTER", 1) and self.is_kw("JOIN", 2))))
            var is_comma = self.is_sym(",")
            if not (is_comma or inner_kw or outer_kw):
                if t.kind == T_IDENT and (t.up == "RIGHT" or t.up == "FULL" or t.up == "CROSS" or t.up == "NATURAL" or
                                          t.up == "ASOF" or t.up == "POSITIONAL" or t.up == "LATERAL" or t.up == "INNER" or t.up == "LEFT" or
                                          t.up == "SEMI" or t.up == "ANTI"):
                    var named = t.up
                    if (t.up == "RIGHT" or t.up == "FULL") and self.is_kw("OUTER", 1):
                        named += " OUTER"
                    raise self.unsupported(t, "`" + named + " JOIN`", "")
                return
            if self.st.has_right and (outer_kw or self.st.join_kind != JK_INNER):
                raise self.unsupported(t, "a `" + ("LEFT" if self.st.join_kind == JK_LEFT else "SEMI" if self.st.join_kind == JK_SEMI else "ANTI" if self.st.join_kind == JK_ANTI else t.up) +
                                       " JOIN` together with a third table", "F-159")
            if is_comma:
                _ = self.advance()
                var r = self.table_ref()
                if self.st.has_right:
                    self.st.extra.append(TableRef(r[0], r[1], r[2], r[3], r[4], self.ref_kid, self.ref_cols.copy()))
                else:
                    self.set_right(r)
                    self.st.comma_join = True
                continue
            if outer_kw:
                if t.up == "LEFT":
                    self.st.join_kind = JK_LEFT
                elif t.up == "SEMI":
                    self.st.join_kind = JK_SEMI
                else:
                    self.st.join_kind = JK_ANTI
                _ = self.advance()
                if self.is_kw("OUTER"):
                    _ = self.advance()
            elif t.up == "INNER":
                _ = self.advance()
            _ = self.advance()                                                  # JOIN
            var r = self.table_ref()
            var more = self.st.has_right
            if more:
                self.st.extra.append(TableRef(r[0], r[1], r[2], r[3], r[4], self.ref_kid, self.ref_cols.copy()))
            else:
                self.set_right(r)
            if self.is_kw("ON"):
                _ = self.advance()
                var on = self.expr()
                if self.st.join_on >= 0:                                        # the ON conditions of an inner join chain are conjuncts of one condition
                    self.st.exprs.append(SqlExpr(X_BIN, "AND", String(), self.st.join_on, on, -1, List[Int](), False, t.line, t.col))
                    self.depths.append(max(self.depths[self.st.join_on], self.depths[on]) + 1)
                    self.st.join_on = len(self.st.exprs) - 1
                else:
                    self.st.join_on = on
            elif self.is_kw("USING"):
                if more:
                    raise self.unsupported(self.tok(), "`USING` in a join of more than two tables", "write ON")
                _ = self.advance()
                _ = self.expect_sym("(")
                while True:
                    var c = self.tok()
                    if c.kind != T_IDENT and c.kind != T_QIDENT:
                        raise self.expected("a column name")
                    self.st.join_using.append(c.text)
                    _ = self.advance()
                    if self.is_sym(","):
                        _ = self.advance()
                        continue
                    break
                _ = self.expect_sym(")")
            else:
                raise self.expected("ON or USING after JOIN")

    def set_right(mut self, r: Tuple[String, String, Bool, Int, Int]):
        self.st.has_right = True
        self.st.right_table = r[0]
        self.st.right_alias = r[1]
        self.st.right_is_file = r[2]
        self.st.right_line = r[3]
        self.st.right_col = r[4]
        self.st.right_kid = self.ref_kid
        self.st.right_cols = self.ref_cols.copy()

    # -- expressions, loosest first
    def expr(mut self) raises -> Int:
        self.enter(self.tok())
        var e = self.or_expr()
        self.level -= 1
        return e

    def or_expr(mut self) raises -> Int:
        var left = self.and_expr()
        while self.is_kw("OR"):
            var t = self.advance()
            var right = self.and_expr()
            left = self.node(X_BIN, "OR", left, right, -1, t)
        return left

    def and_expr(mut self) raises -> Int:
        var left = self.not_expr()
        while self.is_kw("AND"):
            var t = self.advance()
            var right = self.not_expr()
            left = self.node(X_BIN, "AND", left, right, -1, t)
        return left

    def not_expr(mut self) raises -> Int:
        if self.is_kw("NOT") and not self.is_kw("EXISTS", 1):
            var t = self.advance()
            self.enter(t)
            var a = self.not_expr()
            self.level -= 1
            return self.node(X_NOT, "NOT", a, -1, -1, t)
        return self.cmp_expr()

    def cmp_expr(mut self) raises -> Int:
        var left = self.add_expr()
        var t = self.tok()
        if t.kind == T_SYM and (t.text == "=" or t.text == "<>" or t.text == "!=" or t.text == "<" or t.text == "<=" or
                                t.text == ">" or t.text == ">="):
            _ = self.advance()
            if self.is_kw("ANY") or self.is_kw("ALL") or self.is_kw("SOME"):
                raise self.unsupported(self.tok(), "`" + self.tok().up + "` (a quantified comparison)", "F-154")
            var right = self.add_expr()
            return self.node(X_BIN, "<>" if t.text == "!=" else t.text, left, right, -1, t)
        if t.kind == T_IDENT and t.up == "IS":
            _ = self.advance()
            var neg = False
            if self.is_kw("NOT"):
                _ = self.advance()
                neg = True
            if self.is_kw("NULL"):
                _ = self.advance()
                return self.node(X_ISNULL, "", left, -1, -1, t, neg)
            var u = self.tok()
            if u.kind == T_IDENT and u.up == "DISTINCT":
                _ = self.advance()
                _ = self.expect_kw("FROM")
                var right = self.add_expr()
                return self.node(X_DISTINCT, "", left, right, -1, t, neg)
            if u.kind == T_IDENT and (u.up == "TRUE" or u.up == "FALSE"):
                _ = self.advance()
                return self.node(X_ISBOOL, u.up, left, -1, -1, t, neg)
            if u.kind == T_IDENT and u.up == "UNKNOWN":                  # a boolean that is unknown is NULL
                _ = self.advance()
                return self.node(X_ISNULL, "", left, -1, -1, t, neg)
            raise self.expected("NULL, TRUE, FALSE or DISTINCT FROM after IS")
        var neg = False
        var at = t.copy()
        if t.kind == T_IDENT and t.up == "NOT" and (self.is_kw("BETWEEN", 1) or self.is_kw("IN", 1) or self.is_kw("LIKE", 1) or
                                                   self.is_kw("ILIKE", 1) or self.is_kw("GLOB", 1) or self.is_kw("SIMILAR", 1)):
            neg = True
            _ = self.advance()
            at = self.tok()
        if at.kind == T_IDENT and at.up == "BETWEEN":
            _ = self.advance()
            var lo = self.add_expr()
            _ = self.expect_kw("AND")
            var hi = self.add_expr()
            return self.node(X_BETWEEN, "", left, lo, hi, at, neg)
        if at.kind == T_IDENT and at.up == "IN":
            _ = self.advance()
            _ = self.expect_sym("(")
            if self.is_kw("FROM"):
                raise self.unsupported(self.tok(), "a subquery that starts with `FROM`", "")
            if self.is_kw("SELECT") or self.is_kw("WITH"):
                var si = self.query_after_paren()
                var sid = self.node(X_IN_Q, "", left, -1, -1, at, neg)
                self.st.exprs[sid].items.append(si)
                return sid
            var id = self.node(X_IN, "", left, -1, -1, at, neg)
            while True:
                var item = self.add_expr()
                self.st.exprs[id].items.append(item)
                if self.is_sym(","):
                    _ = self.advance()
                    continue
                break
            _ = self.expect_sym(")")
            return id
        if at.kind == T_IDENT and (at.up == "LIKE" or at.up == "ILIKE"):
            _ = self.advance()
            var pat = self.add_expr()
            var esc = -1
            if self.is_kw("ESCAPE"):
                _ = self.advance()
                esc = self.add_expr()
            return self.node(X_LIKE, at.up, left, pat, esc, at, neg)
        if at.kind == T_IDENT and (at.up == "GLOB" or at.up == "SIMILAR"):
            raise self.unsupported(at, "`" + ("NOT " if neg else "") + at.up + "`", "")
        return left

    def add_expr(mut self) raises -> Int:
        var left = self.mul_expr()
        while self.is_sym("+") or self.is_sym("-") or self.is_sym("||"):
            var t = self.advance()
            var right = self.mul_expr()
            left = self.node(X_BIN, t.text, left, right, -1, t)
        return left

    def mul_expr(mut self) raises -> Int:
        var left = self.unary()
        while self.is_sym("*") or self.is_sym("/") or self.is_sym("//") or self.is_sym("%"):
            var t = self.advance()
            var right = self.unary()
            left = self.node(X_BIN, t.text, left, right, -1, t)
        return left

    def unary(mut self) raises -> Int:
        if self.is_sym("-") or self.is_sym("+"):
            var t = self.advance()
            self.enter(t)
            var a = self.unary()
            self.level -= 1
            if t.text == "+":
                return a
            ref x = self.st.exprs[a]
            if x.kind == X_INT or x.kind == X_DEC or x.kind == X_FLT:            # -5 is one literal
                if x.text.byte_length() > 0 and x.text.as_bytes()[0] != UInt8(45):
                    self.st.exprs[a].text = "-" + x.text
                    self.st.exprs[a].line = t.line
                    self.st.exprs[a].col = t.col
                    return a
            return self.node(X_NEG, "-", a, -1, -1, t)
        return self.postfix()

    def postfix(mut self) raises -> Int:
        var e = self.primary()
        while self.is_sym("::"):
            var t = self.advance()
            e = self.cast_node(e, t, False)
        return e

    def type_spec(mut self) raises -> Tuple[String, Int, Int]:
        """A type name for CAST: its canonical upper-case name, precision and scale (-1 for none)."""
        var t = self.tok()
        if t.kind != T_IDENT:
            raise self.expected("a type name")
        _ = self.advance()
        var w = t.up
        var name = String()
        if w == "INT" or w == "INTEGER" or w == "INT4" or w == "SIGNED":
            name = "INTEGER"
        elif w == "BIGINT" or w == "INT8" or w == "LONG":
            name = "BIGINT"
        elif w == "SMALLINT" or w == "INT2" or w == "SHORT":
            name = "SMALLINT"
        elif w == "TINYINT" or w == "INT1":
            name = "TINYINT"
        elif w == "DECIMAL" or w == "NUMERIC":
            name = "DECIMAL"
        elif w == "DOUBLE" or w == "FLOAT8":
            name = "DOUBLE"
            if w == "DOUBLE" and self.is_kw("PRECISION"):
                _ = self.advance()
        elif w == "FLOAT" or w == "FLOAT4" or w == "REAL":
            name = "REAL"
        elif w == "VARCHAR" or w == "TEXT" or w == "STRING" or w == "CHAR" or w == "BPCHAR":
            name = "VARCHAR"
        elif w == "DATE":
            name = "DATE"
        elif w == "BOOLEAN" or w == "BOOL" or w == "LOGICAL":
            name = "BOOLEAN"
        else:
            raise self.unsupported(t, "the type `" + t.text + "` in a cast (the casts are INTEGER, BIGINT, SMALLINT, TINYINT, DECIMAL(p, s), DOUBLE, REAL, VARCHAR, DATE and BOOLEAN)", "")
        var prec = -1
        var scale = -1
        if self.is_sym("("):
            _ = self.advance()
            var a = self.tok()
            if a.kind != T_INT:
                raise self.expected("a number")
            _ = self.advance()
            prec = Int(a.text)
            if self.is_sym(","):
                _ = self.advance()
                var b = self.tok()
                if b.kind != T_INT:
                    raise self.expected("a number")
                _ = self.advance()
                scale = Int(b.text)
            _ = self.expect_sym(")")
            if name != "DECIMAL" and name != "VARCHAR":
                raise _tok_error(a, "the type " + name + " takes no size")
            if name == "VARCHAR" and scale >= 0:
                raise _tok_error(a, "VARCHAR takes one size")
        return (name, prec, scale)

    def cast_node(mut self, operand: Int, t: Token, try_: Bool) raises -> Int:
        var ty = self.type_spec()
        var id = self.node(X_CAST, ty[0], operand, -1, -1, t, try_)
        self.st.exprs[id].b = ty[1]                                  # (not operands: the precision and scale)
        self.st.exprs[id].c = ty[2]
        return id

    def primary(mut self) raises -> Int:
        var t = self.tok()
        if t.kind == T_INT:
            _ = self.advance()
            return self.node(X_INT, t.text, -1, -1, -1, t)
        if t.kind == T_DEC:
            _ = self.advance()
            return self.node(X_DEC, t.text, -1, -1, -1, t)
        if t.kind == T_FLT:
            _ = self.advance()
            return self.node(X_FLT, t.text, -1, -1, -1, t)
        if t.kind == T_STR:
            _ = self.advance()
            return self.node(X_STR, t.text, -1, -1, -1, t)
        if t.kind == T_SYM:
            if t.text == "(":
                _ = self.advance()
                if self.is_kw("FROM"):
                    raise self.unsupported(self.tok(), "a subquery that starts with `FROM`", "")
                if self.is_kw("SELECT") or self.is_kw("WITH"):
                    var k = self.query_after_paren()
                    var sid = self.node(X_SCALAR_Q, "", -1, -1, -1, t)
                    self.st.exprs[sid].items.append(k)
                    return sid
                var e = self.expr()
                if self.is_sym(","):
                    raise self.unsupported(self.tok(), "a row value `(a, b)`", "")
                _ = self.expect_sym(")")
                return e
            if t.text == "?":
                raise _tok_error(t, "parameters (`?`) are not supported")
            if t.text == "*":
                raise _tok_error(t, "`*` can only be used as COUNT(*)")
            raise self.expected("an expression")
        if t.kind == T_QIDENT:
            _ = self.advance()
            return self.column_rest(t)
        if t.kind == T_IDENT:
            var w = t.up
            if w == "NULL":
                _ = self.advance()
                return self.node(X_NULL, "NULL", -1, -1, -1, t)
            if w == "TRUE":
                _ = self.advance()
                return self.node(X_TRUE, "true", -1, -1, -1, t)
            if w == "FALSE":
                _ = self.advance()
                return self.node(X_FALSE, "false", -1, -1, -1, t)
            if w == "DATE" and self.toks[min(self.pos + 1, len(self.toks) - 1)].kind == T_STR:
                _ = self.advance()
                var s = self.advance()
                return self.node(X_DATE, s.text, -1, -1, -1, t)
            var next_string = self.toks[min(self.pos + 1, len(self.toks) - 1)].kind == T_STR
            var next_paren = self.is_sym("(", 1)
            if w == "CASE":
                return self.case_expr()
            if (w == "CAST" or w == "TRY_CAST") and next_paren:
                _ = self.advance()
                _ = self.advance()
                var operand = self.expr()
                _ = self.expect_kw("AS")
                var cid = self.cast_node(operand, t, w == "TRY_CAST")
                _ = self.expect_sym(")")
                return cid
            if w == "EXTRACT" and next_paren:
                _ = self.advance()
                _ = self.advance()
                var pt = self.tok()
                if pt.kind != T_IDENT and pt.kind != T_STR:
                    raise self.expected("a date part (year, month, day, ...)")
                _ = self.advance()
                _ = self.expect_kw("FROM")
                var arg = self.expr()
                _ = self.expect_sym(")")
                var part = self.node(X_STR, lower_ascii(pt.text), -1, -1, -1, pt)
                var eid = self.node(X_CALL, "EXTRACT", 0, -1, -1, t)
                self.st.exprs[eid].items.append(part)
                self.st.exprs[eid].items.append(arg)
                return eid
            if w == "INTERVAL" and (next_string or self.toks[min(self.pos + 1, len(self.toks) - 1)].kind == T_INT or next_paren):
                return self.interval_expr()
            if (w == "TIMESTAMP" or w == "TIME") and next_string:
                raise self.unsupported(t, "`" + w + "` literals", "omxcore has no timestamp kind")
            if (w == "ARRAY" or w == "STRUCT" or w == "MAP" or w == "LIST") and next_paren:
                raise self.unsupported(t, "`" + w + "` values", "")
            if w == "EXISTS" or (w == "NOT" and self.is_kw("EXISTS", 1)):
                var neg = w == "NOT"
                if neg:
                    _ = self.advance()
                _ = self.advance()
                _ = self.expect_sym("(")
                if not (self.is_kw("SELECT") or self.is_kw("WITH")):
                    raise self.expected("SELECT")
                var si = self.query_after_paren()
                var eid = self.node(X_EXISTS_Q, "", -1, -1, -1, t, neg)
                self.st.exprs[eid].items.append(si)
                return eid
            if _reserved(w) and w != "LEFT" and w != "RIGHT":
                raise self.expected("an expression")
            _ = self.advance()
            if self.is_sym("("):
                return self.call(t)
            return self.column_rest(t)
        raise self.expected("an expression")

    def case_expr(mut self) raises -> Int:
        var t = self.advance()                                                   # CASE
        var operand = -1
        if not self.is_kw("WHEN"):
            operand = self.expr()
        var items = List[Int]()
        while self.is_kw("WHEN"):
            _ = self.advance()
            items.append(self.expr())
            _ = self.expect_kw("THEN")
            items.append(self.expr())
        if len(items) == 0:
            raise self.expected("WHEN")
        var else_ = -1
        if self.is_kw("ELSE"):
            _ = self.advance()
            else_ = self.expr()
        _ = self.expect_kw("END")
        return self.node_items(X_CASE, "", operand, -1, else_, items^, t)

    def interval_expr(mut self) raises -> Int:
        """`INTERVAL 3 DAY`, `INTERVAL '3' DAY`, `INTERVAL '3 days'`, `INTERVAL (n) MONTH`: a number of days, weeks, months or years (an operand of a date's + and -)."""
        var t = self.advance()                                                   # INTERVAL
        var amount = -1
        var unit = String()
        var u = self.tok()
        if u.kind == T_STR:
            _ = self.advance()
            var parts = List[String]()
            for part in u.text.split(" "):
                if part.byte_length() > 0:
                    parts.append(String(part))
            if len(parts) == 0 or len(parts) > 2:
                raise self.unsupported(u, "the interval '" + u.text + "' (write one amount and one unit: INTERVAL '3' MONTH)", "")
            var digits = parts[0]
            var db = digits.as_bytes()
            var ok = len(db) > 0
            for i in range(len(db)):
                if not (_is_digit(db[i]) or (i == 0 and db[i] == UInt8(45) and len(db) > 1)):
                    ok = False
            if not ok:
                raise self.unsupported(u, "the interval amount '" + digits + "' (a whole number)", "")
            amount = self.node(X_INT, digits, -1, -1, -1, u)
            if len(parts) == 2:
                unit = upper_ascii(parts[1])
            else:
                var ut = self.tok()
                if ut.kind != T_IDENT:
                    raise self.expected("a unit (DAY, WEEK, MONTH or YEAR)")
                _ = self.advance()
                unit = ut.up
        else:
            if u.kind == T_INT:
                _ = self.advance()
                amount = self.node(X_INT, u.text, -1, -1, -1, u)
            elif self.is_sym("("):
                _ = self.advance()
                amount = self.expr()
                _ = self.expect_sym(")")
            else:
                raise self.expected("an interval amount (a number, a string such as '3', or an expression in parentheses)")
            var ut = self.tok()
            if ut.kind != T_IDENT:
                raise self.expected("a unit (DAY, WEEK, MONTH or YEAR)")
            _ = self.advance()
            unit = ut.up
        if unit.byte_length() > 1 and unit.as_bytes()[unit.byte_length() - 1] == UInt8(83):         # DAYS -> DAY
            var cut = unit.byte_length() - 1
            var trimmed = String(unit[byte=0:cut])
            unit = trimmed^
        if unit != "DAY" and unit != "WEEK" and unit != "MONTH" and unit != "YEAR":
            raise self.unsupported(t, "an interval in " + unit.lower() + "s", "omxcore has no timestamp kind: days, weeks, months and years only")
        return self.node(X_INTERVAL, unit, amount, -1, -1, t)

    def column_rest(mut self, first: Token) raises -> Int:
        if self.is_sym("."):
            _ = self.advance()
            if self.is_sym("*"):
                raise self.unsupported(self.tok(), "`" + first.text + ".*` outside the select list", "")
            var nm = self.tok()
            if nm.kind != T_IDENT and nm.kind != T_QIDENT:
                raise self.expected("a column name after `.`")
            _ = self.advance()
            if self.is_sym("."):
                raise self.unsupported(self.tok(), "a qualified name with more than two parts", "")
            var id = self.node(X_COL, nm.text, -1, -1, -1, first)
            self.st.exprs[id].text2 = first.text
            return id
        return self.node(X_COL, first.text, -1, -1, -1, first)

    def call(mut self, name: Token) raises -> Int:
        _ = self.advance()                                                      # (
        var id = self.node(X_CALL, name.up, 0, -1, -1, name)
        if self.is_sym(")"):
            _ = self.advance()
        else:
            if self.is_kw("DISTINCT"):
                _ = self.advance()
                self.st.exprs[id].flag = True
            if self.is_sym("*"):
                _ = self.advance()
                self.st.exprs[id].a = 1
            else:
                while True:
                    var arg = self.expr()
                    self.st.exprs[id].items.append(arg)
                    if self.is_sym(","):
                        _ = self.advance()
                        continue
                    if (name.up == "SUBSTRING" or name.up == "SUBSTR") and len(self.st.exprs[id].items) == 1 and self.is_kw("FROM"):
                        _ = self.advance()                                       # SUBSTRING(s FROM 2 FOR 3)
                        var from_e = self.expr()
                        self.st.exprs[id].items.append(from_e)
                        if self.is_kw("FOR"):
                            _ = self.advance()
                            var for_e = self.expr()
                            self.st.exprs[id].items.append(for_e)
                    break
            if self.is_kw("ORDER"):
                raise self.unsupported(self.tok(), "`ORDER BY` inside " + name.up + "(...)", "")
            _ = self.expect_sym(")")
        if self.is_kw("IGNORE") or self.is_kw("RESPECT"):
            raise self.unsupported(self.tok(), "`" + self.tok().up + " NULLS`", "")
        if self.is_kw("OVER"):
            _ = self.advance()
            var w = self.window_spec()
            self.st.exprs[id].c = w
        if self.is_kw("FILTER"):
            raise self.unsupported(self.tok(), "`FILTER`", "")
        if self.is_kw("WITHIN"):
            raise self.unsupported(self.tok(), "`WITHIN GROUP`", "")
        return id


def parse_query(text: String) raises -> SqlQuery:
    """The syntax trees of one query (SELECT, EXPLAIN SELECT, WITH, UNION ...): the statements it is made of, the one the text ends with last."""
    var toks = lex_sql(text)
    if len(toks) == 1:
        raise sql_error(1, 1, "the statement is empty")
    var p = _Parser(toks^, text)
    p.statement()
    var q = SqlQuery()
    swap(q.stmts, p.store)
    var root = SqlStatement(text)
    swap(root, p.st)
    q.stmts.append(root^)
    return q^


def parse_sql(text: String) raises -> SqlStatement:
    """The syntax tree of one plain SELECT or EXPLAIN SELECT statement (no WITH, subquery, derived table or set operation: `parse_query` reads those);
    raises `SQL error at line L, column C: ...`."""
    var q = parse_query(text)
    if len(q.stmts) != 1:
        raise sql_error(1, 1, "this query has a WITH clause, a subquery or a UNION: parse_query reads those")
    return q.stmts.pop()


def query_files(q: SqlQuery) -> List[String]:
    """The file names ('orders.parquet') the query reads in any FROM, once each (the statements in the order of the query's store: a subquery before the statement that holds it)."""
    var out = List[String]()
    for i in range(len(q.stmts)):
        ref st = q.stmts[i]
        if st.table_is_file:
            out.append(st.table)
        if st.has_right and st.right_is_file:
            out.append(st.right_table)
        for ex in st.extra:
            if ex.is_file:
                out.append(ex.name)
    var seen = List[String]()
    for f in out:
        var dup = False
        for g in seen:
            if g == f:
                dup = True
        if not dup:
            seen.append(f)
    return seen^
