import sqlite3, re, math, hashlib
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pyautogui
import time
import pytesseract
from PIL import ImageGrab
from pynput import keyboard, mouse
import threading
# ---------------- Funções auxiliares ----------------

DB_PATH = "modo_aprender.db"  # banco do seu coletor

def carregar_eventos(limit=50000):
    """Lê eventos do banco de dados do modo_aprender."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, janela, tipo, detalhe FROM logs ORDER BY id ASC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def one_hot(idx, dim):
    v = np.zeros((dim,), dtype=np.float32)
    v[idx] = 1.0
    return v

def tempo_features(ts):
    """Converte timestamp em seno/cosseno (hora e dia da semana)."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    h = ts.hour + ts.minute/60.0
    dow = ts.weekday()
    return np.array([
        math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
        math.sin(2*math.pi*dow/7), math.cos(2*math.pi*dow/7)
    ], dtype=np.float32)

def mouse_features(det, win_w=1920, win_h=1080):
    """Extrai coordenadas de clique normalizadas, se existirem."""
    m = re.search(r"\((\d+),\s*(\d+)\)", det)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return np.array([x/win_w, y/win_h], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)

def hash_word(word, dim=256):
    """Hash determinístico de uma palavra para índice fixo."""
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
    return h % dim

def ocr_embedding(texto, dim=256):
    """Transforma OCR em vetor fixo via hashing trick + log-freq."""
    words = [w.lower() for w in re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", texto)]
    vec = np.zeros(dim, dtype=np.float32)
    for w in words:
        idx = hash_word(w, dim)
        vec[idx] += 1
    vec = np.log1p(vec)
    if vec.sum() > 0:
        vec /= np.linalg.norm(vec)
    return vec

def tokenizar_evento(typ, det):
    """Cria um rótulo simbólico para cada tipo de evento."""
    if typ == "KeyPress":
        if len(det) == 1:
            return f"key:{det.lower()}"
        if det.lower().startswith(("ctrl+", "alt+", "shift+")):
            return f"shortcut:{det.lower()}"
        return "key:other"
    if typ == "Clique":
        side = "left" if "left" in det else "right" if "right" in det else "unk"
        return f"click:{side}"
    if typ == "OCR":
        words = [w.lower() for w in re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", det)]
        return "ocr:" + ("|".join(words[:5]) if words else "noocr")
    return "other"

# ---------------- Construção do dataset ----------------

def construir_dataset(seq_len=20, step=1, limit=50000, vocab_top=2000, ocr_dim=256):
    eventos = carregar_eventos(limit)

    action_counts = Counter()
    window_counts = Counter()
    items = []
    for ts, win, typ, det in eventos:
        a = tokenizar_evento(typ, det)
        action_counts[a] += 1
        window_counts[win] += 1
        items.append((ts, win, typ, det, a))

    # vocabulário ações
    top_actions = {a for a,_ in action_counts.most_common(vocab_top)}
    action2id = defaultdict(lambda: len(top_actions))
    for i,a in enumerate(sorted(top_actions)):
        action2id[a] = i
    action_dim = len(top_actions)+1

    # vocabulário janelas
    top_windows = {w for w,_ in window_counts.most_common(1000)}
    win2id = defaultdict(lambda: len(top_windows))
    for i,w in enumerate(sorted(top_windows)):
        win2id[w] = i
    win_dim = len(top_windows)+1

    # monta vetores
    X_raw, Y_ids = [], []
    for ts, win, typ, det, act in items:
        feat = []
        feat.extend(one_hot(win2id[win], win_dim))           # janela
        feat.extend(one_hot(action2id[act], action_dim))     # ação
        feat.extend(tempo_features(ts))                      # tempo
        feat.extend(mouse_features(det) if typ=="Clique" else [0,0])  # clique
        feat.extend(ocr_embedding(det, dim=ocr_dim) if typ=="OCR" else np.zeros(ocr_dim, dtype=np.float32))
        X_raw.append(np.array(feat, dtype=np.float32))
        Y_ids.append(action2id[act])

    X_raw = np.array(X_raw)
    in_dim = X_raw.shape[1]

    # sequência temporal
    X_seq, Y = [], []
    for t in range(seq_len, len(X_raw)-1, step):
        X_seq.append(X_raw[t-seq_len:t])
        Y.append(Y_ids[t])

    return np.array(X_seq), np.array(Y), in_dim, action_dim, dict(action2id), dict(win2id)

# ---------------- ESN (Reservatório caótico) ----------------

def zscore(X, eps=1e-8):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    return (X - m) / (s + eps)

class ESN:
    def __init__(self, in_dim, res_dim=500, leak_rate=0.3,
                 spectral_radius=1.2, density=0.05,
                 in_scale=1.0, ridge=1e-2, seed=42, out_dim=10):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.res_dim = res_dim
        self.out_dim = out_dim
        self.leak = leak_rate
        self.ridge = ridge

        # Pesos de entrada
        self.W_in = (rng.standard_normal((res_dim, in_dim)) * in_scale).astype(np.float32)

        # Reservatório esparso
        W = np.zeros((res_dim, res_dim), dtype=np.float32)
        nnz = int(density * res_dim * res_dim)
        idx = rng.choice(res_dim*res_dim, size=nnz, replace=False)
        W.flat[idx] = rng.standard_normal(nnz).astype(np.float32)

        # Ajusta raio espectral (edge of chaos)
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals)) + 1e-8
        W *= (spectral_radius / rho).astype(np.float32)
        self.W = W

        # Readout
        self.W_out = np.zeros((out_dim, res_dim), dtype=np.float32)
        self.b_out = np.zeros((out_dim,), dtype=np.float32)
        self.mean = None
        self.std = None

    def _step(self, x_t, h):
        pre = self.W_in @ x_t + self.W @ h
        h_tilde = np.tanh(pre)
        return (1 - self.leak) * h + self.leak * h_tilde

    def collect_states(self, X_seq):
        # X_seq: [N, T, in_dim]
        N, T, _ = X_seq.shape
        H = np.zeros((N, self.res_dim), dtype=np.float32)
        for t in range(T):
            Xt = X_seq[:, t, :]
            pre = Xt @ self.W_in.T + H @ self.W.T
            H_tilde = np.tanh(pre)
            H = (1 - self.leak) * H + self.leak * H_tilde
        return H

    def fit(self, X_seq, Y_ids, num_classes):
        # Coleta estados do reservatório
        H = self.collect_states(X_seq)   # [N, R]

        # Calcula e armazena a normalização
        self.mean = H.mean(axis=0, keepdims=True)
        self.std = H.std(axis=0, keepdims=True)
        Hn = (H - self.mean) / (self.std + 1e-8)

        # One-hot dos rótulos
        Y = np.zeros((len(Y_ids), num_classes), dtype=np.float32)
        Y[np.arange(len(Y_ids)), Y_ids] = 1.0  # [N, C]

        lam = self.ridge

        # Ridge regression no espaço do reservatório
        A = Hn.T @ Hn + lam * np.eye(Hn.shape[1], dtype=np.float32)  # [R,R]
        B = Hn.T @ Y  # [R,C]

        W_out = np.linalg.solve(A, B)  # [R,C]

        self.W_out = W_out.T  # [C,R]
        self.b_out = (Y.mean(0) - (self.W_out @ Hn.mean(0))).astype(np.float32)

    def predict_proba(self, X_seq):
        if self.mean is None or self.std is None:
            raise RuntimeError("O modelo deve ser treinado antes de prever.")

        H = self.collect_states(X_seq)
        Hn = (H - self.mean) / (self.std + 1e-8)
        logits = (self.W_out @ Hn.T).T + self.b_out
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

# ---------------- Treino ESN + análise ----------------

def treinar_exemplo(seq_len=20, step=1, limit=50000, res_dim=600, ocr_dim=256):
    X, Y, in_dim, action_dim, action2id, win2id = construir_dataset(
        seq_len=seq_len, step=step, limit=limit, ocr_dim=ocr_dim
    )
    if len(X) < 200:
        raise RuntimeError("Poucos dados. Deixe o coletor rodar mais um pouco e tente de novo.")

    # split treino/teste
    n = len(X)
    n_tr = int(0.8 * n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    tr, te = idx[:n_tr], idx[n_tr:]

    esn = ESN(in_dim, res_dim=res_dim, spectral_radius=1.1,
              leak_rate=0.25, density=0.04, ridge=1e-1, out_dim=action_dim)
    esn.fit(X[tr], Y[tr], num_classes=action_dim)

    # avaliação
    P = esn.predict_proba(X[te])
    yhat = P.argmax(1)
    acc = (yhat == Y[te]).mean()

    # top-3 accuracy
    top3 = (np.argsort(-P, axis=1)[:, :3] == Y[te, None]).any(axis=1).mean()

    id2action = {v: k for k, v in action2id.items()}
    exemplos = [(id2action[int(Y[te][i])], id2action[int(yhat[i])]) for i in range(min(10, len(te)))]

    # análise extra OCR
    eventos = carregar_eventos(limit)
    ocr_words = []
    for _, _, typ, det in eventos:
        if typ == "OCR":
            ocr_words.extend([w.lower() for w in re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", det)])
    top_ocr = Counter(ocr_words).most_common(15)

    return {
        "acc_teste": float(acc),
        "top3_acc": float(top3),
        "exemplo_preds": exemplos,
        "top_ocr_words": top_ocr,
        "modelo": esn,
        "id2action": id2action
    }

# ---------------- Visualização embeddings OCR ----------------

def visualizar_embeddings_ocr(palavras, dim=256, metodo="pca", n_most_common=100):
    counts = Counter(palavras).most_common(n_most_common)
    termos = [w for w,_ in counts]
    freqs = [c for _,c in counts]

    X = np.array([ocr_embedding(w, dim=dim) for w in termos])

    if metodo == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    X2d = reducer.fit_transform(X)

    plt.figure(figsize=(10,8))
    plt.scatter(X2d[:,0], X2d[:,1], s=np.log1p(freqs)*20, alpha=0.7)
    for i, termo in enumerate(termos):
        plt.text(X2d[i,0]+0.01, X2d[i,1]+0.01, termo, fontsize=8)
    plt.title(f"Visualização embeddings OCR ({metodo.upper()})")
    plt.show()

# ---------------- Main ----------------

if __name__ == "__main__":
    init_db()
    info = treinar_exemplo(limit=80000, res_dim=800, ocr_dim=256)
    print("📊 Acurácia de teste:", info["acc_teste"])
    print("🎯 Top-3 accuracy:", info["top3_acc"])
    print("\n🔮 Algumas previsões (verdadeiro -> predito):")
    for v, p in info["exemplo_preds"]:
        print("   ", v, "->", p)

    print("\n📖 Palavras OCR mais frequentes:")
    for w, c in info["top_ocr_words"]:
        print("   ", w, ":", c)

    eventos = carregar_eventos(50000)
    palavras_ocr = []
    for _, _, typ, det in eventos:
        if typ == "OCR":
            palavras_ocr.extend([w.lower() for w in re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", det)])
    visualizar_embeddings_ocr(palavras_ocr, dim=256, metodo="pca", n_most_common=100)

    # Para iniciar a coleta de dados, descomente a linha abaixo
    # iniciar_modo_escuta()

# ------------------Modo piloto automatico------------------

#------------------Modo gamer------------------

# ---------------- Configuração do banco de dados ----------------

def init_db():
    """Cria a tabela de logs se ela não existir."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, timestamp TEXT, janela TEXT, tipo TEXT, detalhe TEXT)")

def log_event(tipo, detalhe, janela="desconhecida"):
    ts = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO logs (timestamp, janela, tipo, detalhe) VALUES (?, ?, ?, ?)", (ts, janela, tipo, detalhe))

def on_press(key):
    log_event("KeyPress", str(key).strip("'"))  # Ex.: 'a' em vez de "pressionado: 'a'"

def on_click(x, y, button, pressed):
    if pressed:
        side = 'left' if button == mouse.Button.left else 'right' if button == mouse.Button.right else 'unk'
        log_event("Clique", f"{side} ({x}, {y})")

def ocr_loop():
    while True:
        try:
            img = ImageGrab.grab(bbox=(0,0,1920,1080))
            texto = pytesseract.image_to_string(img)
            if texto.strip():
                log_event("OCR", texto)
        except Exception as e:
            print(f"Erro no loop de OCR: {e}")
        time.sleep(5)

def iniciar_modo_escuta():
    """Inicializa e inicia os listeners de teclado, mouse e o loop de OCR."""
    print("Iniciando modo de escuta. Pressione Enter nesta janela para parar.")

    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click)

    keyboard_listener.start()
    mouse_listener.start()

    ocr_thread = threading.Thread(target=ocr_loop, daemon=True)
    ocr_thread.start()

    input("Pressione Enter para parar...\n")

    keyboard_listener.stop()
    mouse_listener.stop()
    print("Listeners parados.")


def get_current_features(win2id, action2id, ocr_dim):  # Adicione isso
    # --- ATENÇÃO: Esta função é um placeholder e não está funcional. ---
    # Para funcionar, precisa de uma biblioteca como 'pygetwindow' para
    # obter a janela ativa e o texto dela (via OCR ou APIs de acessibilidade).
    ts = datetime.now()
    win = "placeholder_window"  # Ex: gw.getActiveWindow().title
    typ = "placeholder_type"    # O tipo de evento teria que ser determinado.
    det = ""                    # O detalhe (texto de OCR, etc) teria que ser capturado.
    act = tokenizar_evento(typ, det)
    feat = []
    feat.extend(one_hot(win2id.get(win, len(win2id)), len(win2id) + 1))
    feat.extend(one_hot(action2id.get(act, len(action2id)), len(action2id) + 1))
    feat.extend(tempo_features(ts))
    feat.extend(mouse_features(det) if typ == "Clique" else [0, 0])
    feat.extend(ocr_embedding(det, ocr_dim) if typ == "OCR" else np.zeros(ocr_dim))
    return np.array(feat, dtype=np.float32)

def reproduzir_sequencia(esn, id2action, sequencia_inicial, num_passos=10, win2id=None, action2id=None, ocr_dim=256):
    # --- ATENÇÃO: O modo piloto automático é um protótipo e não está funcional. ---
    # A função `get_current_features` precisaria ser implementada para capturar
    # o estado real da tela e da janela para que o ciclo de previsão e ação funcione.
    X_seq = np.array([sequencia_inicial])  # [1, seq_len, in_dim]
    current_seq = list(sequencia_inicial)  # Lista para append/pop
    for _ in range(num_passos):
        proba = esn.predict_proba(X_seq)
        acao_id = np.argmax(proba[0])
        acao = id2action.get(acao_id, "other")
        print(f"Executando ação prevista: {acao}")
        if "key:" in acao:
            pyautogui.press(acao.split(":")[1])
        elif "click:" in acao:
            # Para um clique funcional, as coordenadas precisariam ser extraídas
            # do evento previsto e passadas para a função de clique.
            pyautogui.click()
        time.sleep(0.1)

        # A linha abaixo é a principal razão pela qual o piloto não funciona.
        # `get_current_features` não captura o estado real do sistema.
        new_feat = get_current_features(win2id, action2id, ocr_dim)
        current_seq.append(new_feat)
        current_seq.pop(0)  # Manter seq_len
        X_seq = np.array([current_seq])
