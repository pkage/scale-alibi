
const easeInOutCubic = x => {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2
}

const sleep = ms => new Promise( (resolve, _) => {
    setTimeout( resolve, ms )
})

const scrollTo = async (element, to, duration) => {
    if (duration <= 0) return
    
    const startpos = element.scrollTop
    const difference = to - element.scrollTop


    for (let i = 0; i <= 1; i += 0.01) {
        element.scrollTop = startpos + (difference * easeInOutCubic(i))
        await sleep(duration/100)
    }
}


// lifted directly from the docs, should redo this later on
document.addEventListener('DOMContentLoaded', async () => {

    // add scrolls
    document.querySelectorAll('[data-scrollto]').forEach(el => {
        el.addEventListener('click', async e => {
            e.preventDefault()
            const target_el = document.querySelector(`#${el.dataset.scrollto}`)
            await scrollTo(document.documentElement, target_el.offsetTop - 20, 100)
        })
    })

})


