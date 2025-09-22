  Automatisierte Gebäudeschadensklassifikation (CNN-basiert)

Dieses Repository enthält den Code und ein kleines Beispieldatenset  
für mein Bachelorprojekt „Automatisierte Klassifikation von Gebäudeschäden  
mittels Machine Learning auf Grundlage von Satellitenbildern am Beispiel von Damaskus“.

  Überblick
Ziel des Projekts ist es, den Zerstörungsgrad von Gebäuden automatisch zu schätzen,  
indem **Vorher- und Nachher-Satellitenbilder miteinander verglichen werden.  
Dazu werden die Bilder in 256×256 Pixel große Abschnitte zerlegt  
und durch ein Twin-EfficientNet-B0-Modell ausgewertet.  
Das Ergebnis sind Heatmaps, die den Schaden visuell darstellen,  
sowie ein automatisch erstellter PDF-Bericht mit Kennzahlen und Vorher-Nachher-Vergleichen.

 Inhalte
- full_pipeline.py – Hauptskript (Daten laden → Vorverarbeitung → Modellvorhersage)
- heatmap_generator_v1.py – Erzeugt Heatmaps und den PDF-Bericht
- auto_generated_labels.csv – Beispiel-Labels zum Testen

 Voraussetzungen
- Python 3.10+
- TensorFlow 2.x  
- NumPy, Pandas, Matplotlib, Pillow  

Installation der benötigten Pakete:
```bash
pip install tensorflow numpy pandas matplotlib pillow
