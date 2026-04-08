# ✈️ Aircraft Damage Analyzer

## 📌 Description

Deep learning project for aircraft damage detection and analysis.
The system classifies aircraft damage types (**dent, crack, scratch, corrosion**) and generates descriptive captions using a vision-language model.

---

## 🚀 Features

* Multi-class image classification using **VGG16**
* Image captioning with **BLIP (HuggingFace)**
* Simple web interface built with **Flask**
* Upload image and get prediction + caption instantly

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* HuggingFace Transformers (BLIP)
* Flask
* HTML / CSS

---

## 📊 Results

* **Test Accuracy:** ~80%
* Output example:

  * Damage Type: `dent`
  * Caption: *"An aircraft surface with a visible dent on the panel"*

---

## 💻 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
```

### 3. Open in browser

```
http://127.0.0.1:5000/
```

Upload an image and get results instantly.

---

## 📂 Project Structure

```
aircraft-damage-analyzer/
├── app.py
├── train.py
├── inference.py
├── blip_model.py
├── requirements.txt
├── README.md
│
├── model/
│   └── classifier.keras
│
├── static/
│   ├── style.css
│   └── uploads/
│
└── templates/
    └── index.html
```

---

## 🧠 Models

* **VGG16** (pre-trained on ImageNet) for classification
* **BLIP** for image captioning

---

## 📄 License

MIT License
