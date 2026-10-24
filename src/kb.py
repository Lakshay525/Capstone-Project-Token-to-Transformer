"""Knowledge-base loading, section-level chunking and hybrid retrieval.

Chunks are the `## [SECTION-ID] Title` sections of kb/*.md (header metadata kept), plus one
chunk per catalog row of data/products.csv (id `products.csv:SKU`). Search is BM25 over
lower-cased word tokens (Latin + Devanagari), with:
  * a small romanised-Hindi/Hinglish -> English lexicon, so "wapas"/"वापस" also matches "return"
    (customers don't use the documents' words);
  * exact-SKU lookup for catalog questions;
  * precedence re-weighting from KB-00.1 (policy > agent guide > help > FAQ > marketing) and a
    penalty for `status: superseded` documents.
An optional multilingual embedding model (sentence-transformers) is fused in with reciprocal
rank fusion when it is installed and EMBED_MODEL is set.
"""
import csv
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import DATA_DIR, KB_DIR

TOKEN_RE = re.compile(r"[a-z0-9ऀ-ॿ]+")
SECTION_RE = re.compile(r"^##\s+\[([A-Za-z0-9.\-]+)\]\s*(.*)$")
SKU_RE = re.compile(r"\b([A-Z]{3}-\d{4})\b", re.I)
STOP = set("the a an of to is are was be in on for and or it my i me can do does you your with at as by "
           "this that if from have has what how when will so not no yes there their they we our".split())

# precedence from KB-00.1 -> multiplicative weight
DOC_TYPE_WEIGHT = {"policy": 1.0, "agent_guide": 0.95, "help": 0.9, "faq": 0.8,
                   "marketing": 0.6, "meta": 0.9, "catalog": 1.0}
SUPERSEDED_WEIGHT = 0.7

# romanised / native Hindi (and common spoken) words -> English terms used in the documents.
LEXICON = {
    "wapas": "return", "vapas": "return", "waapas": "return", "लौटाने": "return", "लौटा": "return",
    "वापस": "return", "लौटाना": "return", "लौटाऊं": "return", "send back": "return",
    "paisa": "refund money", "पैसे": "refund money", "पैसा": "refund money",
    "rifund": "refund", "रिफ़ंड": "refund", "रिफंड": "refund", "refund": "refund",
    "kab": "when how long", "kitne": "how many how long", "kitna": "how much", "कितने": "how many days",
    "din": "days", "दिन": "days", "tak": "within", "kaise": "how", "kya": "",
    "order": "order", "ऑर्डर": "order", "आर्डर": "order",
    "cancel": "cancel cancellation", "कैंसिल": "cancel cancellation", "radd": "cancel",
    "shipped": "shipped ship", "shipping": "delivery", "शिप": "shipped",
    "delivery": "delivery", "डिलीवरी": "delivery", "deliver": "delivery",
    "kharab": "damaged defective", "tuta": "damaged", "toota": "damaged", "टूटा": "damaged",
    "खराब": "damaged defective", "गलत": "wrong", "galat": "wrong",
    "shikayat": "complaint report", "शिकायत": "complaint report", "फ़ोटो": "photos", "फोटो": "photos",
    "ghante": "hours", "घंटे": "hours", "कामकाजी": "business", "kaamkaaji": "business",
    "pickup": "reverse pickup", "पिकअप": "reverse pickup", "collect": "reverse pickup pickup",
    "nobody came": "missed pickup", "did not come": "missed pickup",
    "mufti": "free", "muft": "free", "मुफ़्त": "free", "free": "free fee",
    "fees": "fee", "shulk": "fee", "फ़ीस": "fee", "फीस": "fee",
    "warranty": "warranty", "वारंटी": "warranty", "guarantee": "warranty",
    "coupon": "coupon", "कूपन": "coupon", "voucher": "coupon compensation wallet credit",
    "discount": "coupon offers", "student": "coupon offers discount",
    "wallet": "wallet", "वॉलेट": "wallet", "cod": "cash on delivery", "कैश": "cash",
    "sil": "seal", "सील": "seal sealed", "खोल": "opened", "khol": "opened", "khula": "opened",
    "insaan": "human colleague", "इंसान": "human colleague", "manushya": "human",
    "agent": "human colleague", "person": "human colleague", "बात": "talk",
    "kharidi": "bought", "bheja": "delivered", "pahunch": "delivered", "mila": "received",
    "nahi": "not", "nahin": "not", "late": "late delay", "deri": "late delay", "dair": "late delay",
    "sale": "festive sale", "festive": "festive sale", "tyohar": "festive",
    "plus": "plus membership", "member": "plus membership", "सदस्य": "plus membership",
    "jaldi": "express fast", "express": "express",
    "pincode": "pincode serviceable", "पिनकोड": "pincode serviceable", "dubai": "outside india",
    "abroad": "outside india", "international": "outside india",
    "assured": "assured", "ऐश्योर्ड": "assured", "claim": "claim warranty",
    "bank": "bank transfer", "बैंक": "bank transfer", "upi": "upi", "card": "card",
    "netbanking": "net banking", "net banking": "net banking",
    "address": "address change", "pata": "address", "पता": "address",
    "tracking": "track order status", "track": "track order status", "kahan": "track status where",
    "कहाँ": "track status where", "kahaan": "track status where",
    "speaker": "bluetooth speaker", "earbuds": "wireless earbuds", "smartwatch": "smartwatch",
    "charge": "charging", "charging": "charging",
    "gst": "invoice", "invoice": "invoice", "bill": "invoice",
    "ek": "one", "coupons": "coupon", "do": "two", "teen": "three",
}


# phrase-level bridges from how customers talk to how the documents talk (regex -> doc terms)
PHRASES = [
    (r"never placed|did ?n.t place|did not place|not placed by me|did not make|didn.t make|unauthori[sz]ed|"
     r"not mine|fraud|hacked|someone else.s order|मैंने नहीं", "unauthorised order account activity not made by customer escalate human"),
    (r"crack|broke|broken|torn|shattered|dead on arrival|damag|defect|टूट|फट", "damaged defective item report 48 hours photos"),
    (r"lawyer|legal|sue |court|police|consumer forum|वकील|कोर्ट|पुलिस", "legal action consumer court police complaint escalate human"),
    (r"human|real person|someone|manager|supervisor|representative|colleague|insaan|इंसान|किसी से बात", "customer asks for human agent escalate"),
    (r"stop (it|my order|the order)|on its way|in transit|already (shipped|dispatched)|रोक", "shipped order cannot be cancelled refuse parcel"),
    (r"doesn.t fit|too (small|big|tight)|don.t like|do not like|changed my mind|not needed|no longer need|fit nahi|पसंद नहीं|send (it )?back",
     "change of mind return window"),
    (r"opened|unsealed|tried|worn|used it|खोल", "condition of returned items unused unopened seal"),
    (r"money back|paise (wapas|kab)|पैसे वापस", "refund"),
    (r"how (long|soon|fast)|by when|which date|when (will|do|should)|kab tak|कब तक|कितने दिन", "timeline days"),
    (r"phone number|email of|address of|someone.s (number|details)|another customer|neighbou?r", "never share another customer details read out"),
    (r"cash on delivery|cod|कैश ऑन", "cash on delivery cod refund wallet bank transfer"),
    (r"started on|initiated on|शुरू हुआ|shuru hua", "refund initiated timeline business days method"),
    (r"deliver(ed)? (to|in) \w+|do you deliver|ship(ping)? to|outside india|abroad", "where we deliver only within india pincode serviceable"),
    (r"doesn.?t (work|turn on|power|charge|sync|connect)|won.?t (turn|power|charge|sync|connect)|not (working|charging|syncing)",
     "troubleshooting steps"),
    (r"warrant|claim|guarantee", "warranty assured brand"),
    (r"swimsuit|swimwear|briefs|innerwear|lingerie|bra|underwear", "innerwear not returnable"),
    (r"grocery|vegetable|fruit|milk|perishable|expired", "grocery not returnable damaged expired 48 hours"),
    (r"late|delay|didn.?t arrive|not arrived|hasn.?t arrived|देर", "late delivery promised date compensation"),
    (r"compensat|voucher|goodwill|₹\s?\d{3,}", "compensation not allowed policy escalate"),
    (r"store|shop near|pick ?up from|offline|physical", "pincode delivery only online"),
]


def tokenize(text: str) -> list:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def expand_query(query: str) -> str:
    """Append English document terms for Hinglish / Hindi words found in the query."""
    low = query.lower()
    extra = []
    for tok in TOKEN_RE.findall(low):
        if tok in LEXICON and LEXICON[tok]:
            extra.append(LEXICON[tok])
    for phrase in ("send back", "nobody came", "did not come"):
        if phrase in low:
            extra.append(LEXICON[phrase])
    for pattern, terms in PHRASES:
        if re.search(pattern, low):
            extra.append(terms)
    return query + " " + " ".join(extra) if extra else query


@dataclass
class Chunk:
    id: str                    # e.g. "RET-2" or "products.csv:ELE-0012"
    text: str
    title: str
    doc_id: str
    doc_type: str
    status: str
    effective_from: str
    source: str                # file name
    tokens: list = field(default_factory=list, repr=False)

    def render(self) -> str:
        flag = " [SUPERSEDED - applies only where its successor says so]" if self.status == "superseded" else ""
        head = (f"[{self.id}] {self.title} (doc {self.doc_id}, {self.doc_type}, status {self.status}, "
                f"effective_from {self.effective_from}){flag}")
        return head + "\n" + self.text.strip()


def _parse_header(raw: str):
    meta, body = {}, raw
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, body


def load_kb_chunks() -> list:
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        meta, body = _parse_header(path.read_text(encoding="utf-8"))
        cur_id = cur_title = None
        buf = []

        def flush():
            if cur_id and "".join(buf).strip():
                chunks.append(Chunk(
                    id=cur_id, text="".join(buf), title=cur_title, doc_id=meta.get("doc_id", path.stem),
                    doc_type=meta.get("doc_type", "help"), status=meta.get("status", "current"),
                    effective_from=meta.get("effective_from", ""), source=path.name))

        for line in body.splitlines(keepends=True):
            m = SECTION_RE.match(line)
            if m:
                flush()
                cur_id, cur_title, buf = m.group(1), m.group(2).strip(), []
            elif cur_id:
                buf.append(line)
        flush()
    return chunks


def load_product_chunks() -> list:
    out = []
    with open(DATA_DIR / "products.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            text = (f"SKU {r['sku']}: {r['title']} by {r['brand']}. {r['description']}\n"
                    f"return_category={r['return_category']}; price_inr={r['price_inr']}; "
                    f"fulfilled_by={r['fulfilled_by']}; assured={r['assured']}; "
                    f"warranty_months={r['warranty_months']}")
            out.append(Chunk(id=f"products.csv:{r['sku']}", text=text, title=r["title"], doc_id="CATALOG",
                             doc_type="catalog", status="current", effective_from="", source="products.csv"))
    return out


class BM25:
    def __init__(self, docs_tokens: list, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.len = [len(t) for t in docs_tokens]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query_tokens: list) -> list:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in set(query_tokens):
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class Retriever:
    def __init__(self):
        self.kb = load_kb_chunks()
        self.products = load_product_chunks()
        for c in self.kb + self.products:
            c.tokens = tokenize(f"{c.title} {c.id} {c.text}")
        self.kb_idx = BM25([c.tokens for c in self.kb])
        self.prod_idx = BM25([c.tokens for c in self.products])
        self.by_id = {c.id: c for c in self.kb + self.products}
        self._embed = self._load_embedder()

    # ---- optional dense retrieval -------------------------------------------------------
    def _load_embedder(self):
        name = os.environ.get("EMBED_MODEL")
        if not name:
            return None
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(name)
            mat = model.encode([f"{c.title}. {c.text}" for c in self.kb], normalize_embeddings=True)
            return model, np.asarray(mat)
        except Exception as exc:  # optional dependency; fall back to BM25 alone
            print(f"[kb] embeddings disabled: {exc}")
            return None

    def _dense_rank(self, query: str):
        model, mat = self._embed
        import numpy as np
        q = model.encode([query], normalize_embeddings=True)[0]
        return list(np.argsort(-(mat @ q)))

    # ---- search -------------------------------------------------------------------------
    def search(self, query: str, k: int = 6, k_products: int = 2) -> list:
        """Return a list of (Chunk, score), policy/KB chunks first, then catalog rows."""
        expanded = expand_query(query)
        qt = tokenize(expanded)
        scores = self.kb_idx.scores(qt)
        weighted = []
        for c, s in zip(self.kb, scores):
            w = DOC_TYPE_WEIGHT.get(c.doc_type, 0.8) * (SUPERSEDED_WEIGHT if c.status == "superseded" else 1.0)
            weighted.append(s * w)
        order = sorted(range(len(self.kb)), key=lambda i: -weighted[i])
        if self._embed:  # reciprocal rank fusion with the dense ranking
            dense = self._dense_rank(query)
            rrf = Counter()
            for rank, i in enumerate(order):
                rrf[i] += 1 / (60 + rank)
            for rank, i in enumerate(dense):
                rrf[i] += 1 / (60 + rank)
            order = [i for i, _ in rrf.most_common()]
        hits = [(self.kb[i], weighted[i]) for i in order[:k] if weighted[i] > 0]

        prod_hits = []
        skus = {m.upper() for m in SKU_RE.findall(query)}
        for sku in skus:
            c = self.by_id.get(f"products.csv:{sku}")
            if c:
                prod_hits.append((c, 99.0))
        if len(prod_hits) < k_products:
            ps = self.prod_idx.scores(qt)
            for i in sorted(range(len(ps)), key=lambda i: -ps[i]):
                if len(prod_hits) >= k_products or ps[i] < 6.0:  # only confident title matches
                    break
                if self.products[i] not in [h[0] for h in prod_hits]:
                    prod_hits.append((self.products[i], ps[i]))
        prod_hits = prod_hits[:max(k_products, len(skus))]
        hits = self._add_category_policy(hits, prod_hits)
        hits = self._add_high_value_rule(hits, prod_hits, expanded)
        hits = self._add_precedence_rule(hits)
        return hits + prod_hits

    # return_category (products.csv) -> the policy sections that govern it (RET-2 names the codes)
    CATEGORY_POLICY = {
        "electronics": ["RET-2", "RET-3", "WAR-1"], "fashion": ["RET-2", "RET-8", "RET-6"],
        "home_kitchen": ["RET-2"], "beauty_sealed": ["RET-2", "RET-4"], "books_media": ["RET-2"],
        "grocery": ["RET-2", "RET-5", "DWM-1"], "innerwear": ["RET-2", "RET-5"], "digital": ["RET-2", "RET-5"],
    }

    def _add_category_policy(self, hits: list, prod_hits: list) -> list:
        """Two-source answers: a catalog row tells the category, the policy tells the rule."""
        if not prod_hits:
            return hits
        have = {c.id for c, _ in hits}
        top = prod_hits[0][0]
        m = re.search(r"return_category=(\w+)", top.text)
        if not m:
            return hits
        for cid in self.CATEGORY_POLICY.get(m.group(1), []):
            if cid not in have and cid in self.by_id:
                hits.append((self.by_id[cid], 0.0))
        return hits

    def _add_high_value_rule(self, hits: list, prod_hits: list, expanded: str) -> list:
        """A damaged item priced over Rs 10,000 must be handed over (DWM-3, ESC-1)."""
        if not prod_hits or "damaged defective" not in expand_query(expanded):
            return hits
        m = re.search(r"price_inr=(\d+)", prod_hits[0][0].text)
        if m and int(m.group(1)) > 10000:
            have = {c.id for c, _ in hits}
            hits += [(self.by_id[i], 0.0) for i in ("DWM-3", "ESC-1") if i not in have]
        return hits

    def _add_precedence_rule(self, hits: list) -> list:
        """FAQ/help/marketing text can disagree with policy; attach KB-00.1 so the model sees the order."""
        types = {c.doc_type for c, _ in hits}
        if "policy" in types and types & {"faq", "help", "marketing"} and "KB-00.1" not in {c.id for c, _ in hits}:
            hits.append((self.by_id["KB-00.1"], 0.0))
        return hits

    def get(self, chunk_id: str):
        return self.by_id.get(chunk_id)
