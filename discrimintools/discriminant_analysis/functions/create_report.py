


def create_lda_html(obj):
    """
    Create Linear Discriminant Analysis HTML
    ----------------------------------------
    
    
    """
    def create_html_head(title):
        """Création automatisée du début du fichier html. Incorpore une feuille
        de style CSS populaire (Bootstrap) pour améliorer l'esthétique des
        résultats.
        """
        _head = ("""<!DOCTYPE html>
        <html lang="fr" dir="ltr">
          <head>
            <title>R&#233;sultats : %s</title>
            <meta charset="utf-8" />
            <style></style>
            <link
              rel="stylesheet"
              href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
              integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z"
              crossorigin="anonymous"
            />
          </head>
          <body>
            <div class="container text-center">
                <h2>Proc&#233;dure %s</h2>""") % (title, title)
        return _head

    return NotImplementedError("Not implemented method")