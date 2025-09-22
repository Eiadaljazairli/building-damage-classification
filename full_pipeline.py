import os, json, math, shutil, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

BASE_DIR   = r"C:/Users/eyada/OneDrive/Desktop/Praxis"
CSV_PATH   = os.path.join(BASE_DIR, "auto_generated_labels.csv")
PATCH_DIR  = os.path.join(BASE_DIR, "selected_50_patches_256")

OUT_DIR    = os.path.join(BASE_DIR, "model_outputs")
USE_KERAS_FORMAT = True
MODEL_EXT  = ".keras" if USE_KERAS_FORMAT else ".h5"
MODEL_CKPT = os.path.join(OUT_DIR, f"best_model_ckpt{MODEL_EXT}")
MODEL_PATH = os.path.join(OUT_DIR, f"best_model{MODEL_EXT}")
CAL_PATH   = os.path.join(OUT_DIR, "calibration_linear.json")

PRED_DIR   = os.path.join(BASE_DIR, "predictions_results")
SAVE_PREVIEW_IMAGES = False

IMG_SIZE       = (256, 256)
BATCH_SIZE     = 32
EPOCHS_STAGE1  = 8
EPOCHS_STAGE2  = 6           
LR_STAGE1      = 1e-3
LR_STAGE2      = 1e-4       
VAL_SPLIT      = 0.20
TEST_SPLIT     = 0.10
SEED           = 42
AUTOTUNE       = tf.data.AUTOTUNE
UNFREEZE_LAST  = 20          
ROTATE_MAX_DEG = 0.0         

tf.keras.utils.set_random_seed(SEED)

def _read_csv(csv_path):
    df_raw = pd.read_csv(csv_path)
    orig_cols = list(df_raw.columns)
    lower_map = {c.strip().lower(): c for c in orig_cols}

    aliases = {
        "before_image": ["before_image", "before", "beforepath", "before_path", "b",
                         "img_before", "pre", "pre_image", "image_before", "path_before"],
        "after_image":  ["after_image", "after", "afterpath", "after_path", "a",
                         "img_after", "post", "post_image", "image_after", "path_after"],
        "percent":      ["percent", "label", "y", "damage", "damage_percent",
                         "pct", "percentage", "score", "target", "destroyed", "damage_ratio"],
    }

    def find_col(key):
        for cand in aliases[key]:
            cand_l = cand.strip().lower()
            if cand_l in lower_map:
                return lower_map[cand_l]
        return None

    b_col = find_col("before_image")
    a_col = find_col("after_image")
    p_col = find_col("percent")

    if not (b_col and a_col and p_col):
        raise ValueError(f"CSV missing required columns. Found: {orig_cols}.")

    df = pd.DataFrame({
        "before_image": df_raw[b_col],
        "after_image": df_raw[a_col],
        "percent": df_raw[p_col],
    })

    return df


def _make_paths(df):
    df = df.copy()
    df["before_path"] = df["before_image"].apply(lambda p: str(Path(PATCH_DIR) / p))
    df["after_path"]  = df["after_image"].apply(lambda p: str(Path(PATCH_DIR) / p))
    return df


def _autoclean(df):
    b_ok = df["before_path"].apply(lambda p: Path(p).exists())
    a_ok = df["after_path"].apply(lambda p: Path(p).exists())
    kept = df[b_ok & a_ok].copy()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[AUTOCLEAN] removed {dropped} rows with missing files.")
    return kept

def _read_img_tf(path, img_size_hw):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size_hw)
    img = tf.clip_by_value(img / 255.0, 0.0, 1.0)
    return img


def _same_aug(b, a, label, img_size_hw):
    do_flip = tf.random.uniform([], 0, 1, dtype=tf.float32) > 0.5
    b = tf.cond(do_flip, lambda: tf.image.flip_left_right(b), lambda: b)
    a = tf.cond(do_flip, lambda: tf.image.flip_left_right(a), lambda: a)

    delta = tf.random.uniform([], -0.05, 0.05, dtype=tf.float32)
    b = tf.clip_by_value(b + delta, 0.0, 1.0)
    a = tf.clip_by_value(a + delta, 0.0, 1.0)

    zoom = tf.random.uniform([], 1.0, 1.15, dtype=tf.float32)
    base_size = tf.cast(tf.minimum(img_size_hw[0], img_size_hw[1]), tf.float32)
    crop_size_f = tf.math.round(base_size / zoom)
    crop_size = tf.cast(crop_size_f, tf.int32)

    def _zoom(x):
        h = tf.shape(x)[0]; w = tf.shape(x)[1]
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        y1 = y0 + crop_size
        x1 = x0 + crop_size
        x = x[y0:y1, x0:x1, :]
        x = tf.image.resize(x, img_size_hw)
        return x

    b = _zoom(b); a = _zoom(a)
    return (b, a), label


def _build_ds(df, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, augment=True):
    IMG_T = tf.constant([img_size[0], img_size[1]], dtype=tf.int32)

    def _gen():
        for _, r in df.iterrows():
            yield (r["before_path"], r["after_path"]), np.float32(r["percent"])

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.string)),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )

    def _load(paths, y):
        b = _read_img_tf(paths[0], IMG_T)
        a = _read_img_tf(paths[1], IMG_T)
        return (b, a), y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 1000), reshuffle_each_iteration=True)

    if augment:
        def _wrap(pair, label):
            b, a = pair
            (b2, a2), lbl = _same_aug(b, a, label, IMG_T)
            return (b2, a2), lbl
        ds = ds.map(_wrap, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="EffB0Pre")
class EffB0Preprocess(keras.layers.Layer):
    def call(self, x):
        x255 = x * 255.0
        return keras.applications.efficientnet.preprocess_input(x255)


@register_keras_serializable(package="Ops")
class AbsLayer(keras.layers.Layer):
    def call(self, x):
        return tf.abs(x)

def build_model_pretrained(img_size=(256, 256, 3), unfreeze_last=UNFREEZE_LAST):
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_size, pooling="avg")

    inp_b = keras.Input(img_size, name="before")
    inp_a = keras.Input(img_size, name="after")

    pre = EffB0Preprocess(name="preprocess")
    b_p = pre(inp_b)
    a_p = pre(inp_a)

    fb = base(b_p)
    fa = base(a_p)

    diff = keras.layers.Subtract(name="feat_diff")([fa, fb])
    diff_abs = AbsLayer(name="abs_diff")(diff)
    feat = keras.layers.Concatenate(name="feat_concat")([fa, fb, diff_abs])

    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(feat)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    out = keras.layers.Dense(1, activation="linear", name="percent")(x)

    model = keras.Model([inp_b, inp_a], out, name="TwinEffB0")

    base.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(LR_STAGE1), loss="huber", metrics=["mae"])
    return model, base

def train_and_validate():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    df = _autoclean(_make_paths(_read_csv(CSV_PATH)))

    pmax = float(df["percent"].max())
    label_scale = 100.0 if pmax > 1.5 else 1.0
    print(f"[LABEL] detected scale: 0..{int(label_scale)}")

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test
    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:n_train + n_val].copy()
    df_test  = df.iloc[n_train + n_val:].copy()

    if label_scale == 100.0:
        for d in (df_train, df_val, df_test):
            d["percent"] = d["percent"].astype("float32") / 100.0

    print(f"[DATA] Train={len(df_train)}  Val={len(df_val)}  Test={len(df_test)}")

    ds_tr  = _build_ds(df_train, IMG_SIZE, BATCH_SIZE, shuffle=True,  augment=True)
    ds_val = _build_ds(df_val,   IMG_SIZE, BATCH_SIZE, shuffle=False, augment=False)
    ds_te  = _build_ds(df_test,  IMG_SIZE, BATCH_SIZE, shuffle=False, augment=False)

    model, base = build_model_pretrained(img_size=IMG_SIZE + (3,), unfreeze_last=UNFREEZE_LAST)

    ckpt   = keras.callbacks.ModelCheckpoint(MODEL_CKPT, monitor="val_mae", mode="min", save_best_only=True, verbose=1)
    early  = keras.callbacks.EarlyStopping(monitor="val_mae", patience=6, mode="min", restore_best_weights=True, verbose=1)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=3, mode="min", verbose=1)
    log    = keras.callbacks.CSVLogger(os.path.join(OUT_DIR, "training_log_v5.csv"))

    model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS_STAGE1, callbacks=[ckpt, early, reduce, log], verbose=1)

    base.trainable = True
    for l in base.layers[:-UNFREEZE_LAST]:
        l.trainable = False
    for l in base.layers[-UNFREEZE_LAST:]:
        l.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(LR_STAGE2), loss="huber", metrics=["mae"])
    model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS_STAGE2, callbacks=[ckpt, early, reduce, log], verbose=1)

    model = keras.models.load_model(MODEL_CKPT, compile=False)

    y_true, y_pred = [], []
    for (pair, label) in ds_te:
        pred = model.predict({"before": pair[0], "after": pair[1]}, verbose=0).squeeze()
        y_true.extend(label.numpy().tolist())
        y_pred.extend(np.array(pred).flatten().tolist())
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    def _metrics(y, p):
        mae  = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        r    = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan")
        return mae, rmse, r

    mae_raw, rmse_raw, r_raw = _metrics(y_true, y_pred)
    print(f"[EVAL raw] MAE={mae_raw:.2f}  RMSE={rmse_raw:.2f}  r={r_raw:.3f}")

    yv, pv = [], []
    for (pair, label) in ds_val:
        pred = model.predict({"before": pair[0], "after": pair[1]}, verbose=0).squeeze()
        yv.extend(label.numpy().tolist())
        pv.extend(np.array(pred).flatten().tolist())
    yv = np.array(yv, dtype=np.float32)
    pv = np.array(pv, dtype=np.float32)
    if len(pv) >= 2 and np.std(pv) > 1e-6:
        A = np.vstack([pv, np.ones_like(pv)]).T
        a, b = np.linalg.lstsq(A, yv, rcond=None)[0]
    else:
        a, b = 1.0, 0.0
    with open(CAL_PATH, "w", encoding="utf-8") as f:
        json.dump({"a": float(a), "b": float(b), "label_scale": float(label_scale)}, f, indent=2)
    print(f"[CAL] saved linear calibration: y≈{a:.3f}*pred+{b:.3f} (scale 0..{int(label_scale)})")

    y_pred_cal = a * y_pred + b
    mae_cal, rmse_cal, r_cal = _metrics(y_true, y_pred_cal)
    print(f"[EVAL cal] MAE={mae_cal:.2f}  RMSE={rmse_cal:.2f}  r={r_cal:.3f}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=18, label="raw")
    plt.scatter(y_true, y_pred_cal, s=18, marker="x", label="cal")
    lo = float(np.min([y_true.min(), y_pred.min(), y_pred_cal.min()]))
    hi = float(np.max([y_true.max(), y_pred.max(), y_pred_cal.max()]))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Ground Truth"); plt.ylabel("Prediction")
    plt.title("Calibration: GT vs Pred (raw & linear-cal)")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(OUT_DIR)/"calibration_scatter_v5.png", dpi=200, bbox_inches="tight")
    plt.close()

    try:
        log_df = pd.read_csv(os.path.join(OUT_DIR, "training_log_v5.csv"))
        plt.figure(figsize=(6,4))
        plt.plot(log_df.index + 1, log_df["loss"], label="train loss")
        if "val_loss" in log_df:
            plt.plot(log_df.index + 1, log_df["val_loss"], label="val loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(Path(OUT_DIR)/"training_loss_v5.png", dpi=200, bbox_inches="tight")
        plt.close()
        if "mae" in log_df.columns:
            plt.figure(figsize=(6,4))
            plt.plot(log_df.index + 1, log_df["mae"], label="train mae")
            if "val_mae" in log_df.columns:
                plt.plot(log_df.index + 1, log_df["val_mae"], label="val mae")
            plt.xlabel("Epoch"); plt.ylabel("MAE")
            plt.title("Training MAE")
            plt.legend(); plt.tight_layout()
            plt.savefig(Path(OUT_DIR)/"training_mae_v5.png", dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"[WARN] Could not plot training curves: {e}")

    try:
        resid = y_true - y_pred_cal
        plt.figure(figsize=(6,4))
        plt.hist(resid, bins=20)
        plt.xlabel("Residual (y - yhat_cal)")
        plt.ylabel("Count")
        plt.title("Residuals Histogram (Calibrated)")
        plt.tight_layout()
        plt.savefig(Path(OUT_DIR)/"residuals_hist_v5.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not plot residuals histogram: {e}")

    try:
        shutil.copyfile(MODEL_CKPT, MODEL_PATH)
    except Exception:
        model.save(MODEL_PATH)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = Path(OUT_DIR) / f"best_model_v5_{ts}{MODEL_EXT}"
    try:
        shutil.copyfile(MODEL_CKPT, ts_path)
    except Exception:
        model.save(ts_path)
    print(f"[OK] Best model saved to: {MODEL_PATH} and {ts_path}")

    return model, label_scale, (a, b), df

def _read_img_pil(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = (np.asarray(img).astype("float32") / 255.0)[None, ...]
    return img, arr


def _discover_pairs_from_dir(images_dir):
    before_dir = os.path.join(images_dir, "before")
    after_dir  = os.path.join(images_dir, "after")
    pairs = []

    if os.path.isdir(before_dir) and os.path.isdir(after_dir):
        b_files = sorted([f for f in os.listdir(before_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        a_files = sorted([f for f in os.listdir(after_dir)  if f.lower().endswith((".jpg",".jpeg",".png"))])
        names_b = {os.path.splitext(f)[0] for f in b_files}
        names_a = {os.path.splitext(f)[0] for f in a_files}
        common = sorted(list(names_b & names_a))
        for name in common:
            def pick(d, n):
                for ext in (".jpg", ".jpeg", ".png"):
                    p = os.path.join(d, n + ext)
                    if os.path.exists(p): return p
                return None
            bf, af = pick(before_dir, name), pick(after_dir, name)
            if bf and af: pairs.append((bf, af, name))
        return pairs

    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    befores = [f for f in files if f.lower().startswith("before")]
    afters  = [f for f in files if f.lower().startswith("after")]
    def tail(n, p): return os.path.splitext(n)[0][len(p):]
    map_b = {tail(f, "before"): f for f in befores}
    map_a = {tail(f, "after"):  f for f in afters}
    for t in sorted(set(map_b) & set(map_a)):
        pairs.append((os.path.join(images_dir, map_b[t]), os.path.join(images_dir, map_a[t]), t.strip("_")))
    return pairs


def _discover_pairs_any(df_for_csv):
    pairs = _discover_pairs_from_dir(PATCH_DIR)
    if len(pairs) > 0:
        return pairs
    pairs = []
    for _, r in df_for_csv.iterrows():
        bf = r["before_path"]; af = r["after_path"]
        name = Path(r["before_image"]).stem
        if os.path.exists(bf) and os.path.exists(af):
            pairs.append((bf, af, name))
    return pairs


def predict_and_export(model, label_scale, calib, df_for_csv):
    Path(PRED_DIR).mkdir(parents=True, exist_ok=True)
    out_csv = Path(PRED_DIR) / "results_v5.csv"
    out_img = Path(PRED_DIR) / "images_v5"
    if SAVE_PREVIEW_IMAGES:
        out_img.mkdir(parents=True, exist_ok=True)

    a, b = calib

    pairs = _discover_pairs_any(df_for_csv)
    print(f"[INFO] Found {len(pairs)} pairs for prediction (v5).")

    rows = []
    for bf, af, name in pairs:
        try:
            b_img, b_arr = _read_img_pil(bf)
            a_img, a_arr = _read_img_pil(af)
            pred = float(model.predict({"before": b_arr, "after": a_arr}, verbose=0).reshape(-1)[0])
            pred_cal = float(a * pred + b)

            raw_pct = float(pred * 100.0)
            cal_pct = float(pred_cal * 100.0)

            if SAVE_PREVIEW_IMAGES:
                w, h = b_img.size
                canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
                canvas.paste(b_img, (0, 0)); canvas.paste(a_img, (w, 0))
                txt = f"Pred: {raw_pct:.2f}% | Cal: {cal_pct:.2f}%"
                draw = ImageDraw.Draw(canvas)
                draw.rectangle([5, 5, w * 2 - 5, 36], fill=(255, 255, 255))
                draw.text((10, 10), txt, fill=(0, 0, 0))
                canvas.save(out_img / f"{name}.jpg")

            rows.append({
                "name": name,
                "before_path": bf,
                "after_path": af,
                "pred_raw_percent": round(raw_pct, 4),
                "pred_calibrated_percent": round(cal_pct, 4),
            })
        except Exception as e:
            print(f"[WARN] Skip {name}: {e}")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Predictions saved to {out_csv}")
    if SAVE_PREVIEW_IMAGES:
        print(f"[OK] Images saved to {out_img}")
    else:
        print(f"[OK] (No preview images generated here to keep prediction fast)")

# ============ Main ============
def main():
    model, label_scale, calib, df_for_csv = train_and_validate()
    model = keras.models.load_model(MODEL_CKPT, compile=False)
    predict_and_export(model, label_scale, calib, df_for_csv)
    print("hier nachricht")
print("[DONE] Training + Prediction pipeline (v5) completed.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    main()
