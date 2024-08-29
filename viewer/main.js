import 'leaflet/dist/leaflet.css'
import * as L from 'leaflet'
import * as pmtiles from 'pmtiles'

const LOC_ATLANTA = [33.78660, -84.39112]

const setup_map = () => {
    const map = L.map('map').setView(LOC_ATLANTA, 8)

    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map)

    L.GridLayer.GridDebug = L.GridLayer.extend({
        createTile: function (coords) {
            const tile = document.createElement('div')
            tile.style.outline = '1px solid #009fb7ff'
            tile.style.fontWeight = 'bold'
            tile.style.fontSize = '14pt'
            tile.innerHTML = [coords.z, coords.x, coords.y].join('/');
            return tile;
        },
    })

    L.gridLayer.gridDebug = function (opts) {
        return new L.GridLayer.GridDebug(opts)
    }

    L.gridLayer.gridDebug().addTo(map)


    L.marker(LOC_ATLANTA).addTo(map)
        .bindPopup('edinburgh: 8/125/79')
        // .openPopup()
}


window.addEventListener('load', () => {
    console.log('it begins')

    setup_map()
})
