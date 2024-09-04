import 'leaflet/dist/leaflet.css'
import * as L from 'leaflet'
import { PMTiles, leafletRasterLayer } from 'pmtiles'
import * as dat from 'dat.gui'

const LOC_ATLANTA = [33.78660, -84.39112]

const setup_map = () => {
    const map = L.map('map').setView(LOC_ATLANTA, 8)

    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map)

    const tile_layer = L.layerGroup()
    tile_layer.addTo(map)

    const debug_layer = L.layerGroup()
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

    L.gridLayer.gridDebug().addTo(debug_layer)
    debug_layer.addTo(map)


    L.marker(LOC_ATLANTA).addTo(map)
        .bindPopup('atlanta: 8/125/79')
        // .openPopup()

    return [map, tile_layer, debug_layer]
}


window.addEventListener('load', () => {
    console.log('it begins')

    const [map, tile_layer, debug_layer] = setup_map()

    const tile_files = import.meta.glob('./public/**/*.pmtile')
    
    const tiles = {}
    for (let key in tile_files) {
        let name = key.split('/')
        name = name[name.length - 1]
        tiles[name] = key.replace('public/', '')
    }

    console.log(tiles)

    let obj = {
        active_tile: ''
    }

    var gui = new dat.GUI()
    var folder1 = gui.addFolder('pmtile sources')

    folder1.add(obj, 'active_tile', tiles).onChange(e => {
        debug_layer.removeFrom(map)
        tile_layer.clearLayers()
        const p = new PMTiles(e)
        leafletRasterLayer(p).addTo(tile_layer)
        debug_layer.addTo(map)
    })

})
