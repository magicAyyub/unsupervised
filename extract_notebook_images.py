import os
import json
import base64

def extract_images_from_notebooks(notebooks_dir: str, output_dir: str) -> None:
    # Signature: extract_images_from_notebooks(notebooks_dir: str, output_dir: str) -> None
    # Permet d'extraire toutes les images PNG des cellules des notebooks et de générer un index de référence.
    
    # 1. Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Ouvrir le fichier d'index en écriture
    index_path = os.path.join(output_dir, "index.md")
    with open(index_path, "w", encoding="utf-8") as index_file:
        index_file.write("# Index des Images Extraites\n\n")
        index_file.write("Ce fichier répertorie toutes les figures extraites des notebooks pour faciliter leur intégration dans le rapport.\n\n")
        
        # 3. Lister et trier les notebooks
        notebook_files = sorted([
            f for f in os.listdir(notebooks_dir) 
            if f.endswith(".ipynb") and not f.startswith(".")
        ])
        
        for nb_file in notebook_files:
            nb_path = os.path.join(notebooks_dir, nb_file)
            nb_name = os.path.splitext(nb_file)[0]
            
            index_file.write(f"## Notebook: `{nb_file}`\n\n")
            print(f"Traitement de {nb_file}...")
            
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    notebook = json.load(f)
            except Exception as e:
                print(f"Erreur lors de la lecture de {nb_file}: {e}")
                continue
                
            cells = notebook.get("cells", [])
            image_count = 0
            
            for cell_idx, cell in enumerate(cells):
                cell_type = cell.get("cell_type", "")
                source = "".join(cell.get("source", []))
                
                # Récupérer les outputs si c'est une cellule de code
                outputs = cell.get("outputs", []) if cell_type == "code" else []
                
                for out_idx, output in enumerate(outputs):
                    # Chercher s'il y a des données de type image/png
                    data = output.get("data", {})
                    if "image/png" in data:
                        image_count += 1
                        png_base64 = data["image/png"].replace("\n", "")
                        
                        # Déterminer un nom de fichier propre
                        image_name = f"{nb_name}_cell{cell_idx}_out{out_idx}.png"
                        image_path = os.path.join(output_dir, image_name)
                        
                        # Décoder et sauvegarder l'image
                        try:
                            image_data = base64.b64decode(png_base64)
                            with open(image_path, "wb") as img_f:
                                img_f.write(image_data)
                                
                            # Écrire l'entrée dans l'index avec le contexte (code source ou commentaire)
                            index_file.write(f"### Image {image_name}\n\n")
                            index_file.write(f"![{image_name}]({image_name})\n\n")
                            index_file.write("**Code / Contexte :**\n")
                            index_file.write("```python\n")
                            index_file.write(source[:500] + ("\n... [tronqué]" if len(source) > 500 else ""))
                            index_file.write("\n```\n\n")
                            index_file.write("---\n\n")
                            
                        except Exception as e:
                            print(f"Erreur d'écriture pour {image_name}: {e}")
                            
            print(f"-> {image_count} images extraites de {nb_file}")

if __name__ == "__main__":
    # Définition des chemins de travail
    current_dir = os.path.dirname(os.path.abspath(__file__))
    notebooks_directory = os.path.join(current_dir, "notebooks")
    output_directory = os.path.join(current_dir, "report_assets")
    
    # Lancement de l'extraction
    extract_images_from_notebooks(notebooks_directory, output_directory)
    print("Extraction terminée avec succès. Consultez report_assets/index.md pour l'index.")
