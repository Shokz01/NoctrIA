let themebutton = document.getElementById('theme')
themebutton.addEventListener('click', changetheme)
let body = document.body

let tipo = localStorage.getItem('tipo') === 'true'

if (tipo == true) {
    body.classList.add('claro')
}

function changetheme(){
    if (tipo == false) {
        tipo = true
        body.classList.add('claro')
    } else {
        tipo = false
        body.classList.remove('claro')
    }
    localStorage.setItem('tipo', tipo)
}
