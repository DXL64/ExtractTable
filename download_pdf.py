import gdown


def get_id_from_url(url):
    """Returns the ID from the DefectDojo API.

    :param url: URL returned by the API

    """
    url = url.split('/')
    return url[len(url)-2]


def download_pdf(pdf_link):
    url = 'https://drive.google.com/uc?id='
    id = get_id_from_url(
        pdf_link)
    url = url + id
    output = 'pdf/' + id + '.pdf'
    gdown.download(url, output, quiet=False)
    return output
