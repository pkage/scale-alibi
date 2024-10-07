import 'leaflet/dist/leaflet.css'
import * as L from 'leaflet'
import { PMTiles, leafletRasterLayer } from 'pmtiles'
import * as dat from 'dat.gui'

// const LOC_ATLANTA = [33.78660, -84.39112]
const LOC_CENTER  = [38, -97]

const LAMBDA_BASE = 'https://lmsnjk6vuzd2qw5c5fwm4uwbiy0zlihk.lambda-url.us-east-2.on.aws' // /sign/display/visual_tiles_small_display.pmtile'

const DATASETS = {
    full: {
        radar: 'display/sar_tiles_display.pmtile',
        lores: 'display/visual_tiles_display.pmtile',
        hires: 'display/hires_visual_tiles_display.pmtile'
    },
    small: {
        radar: 'display/sar_tiles_small_display.pmtile',
        lores: 'display/visual_tiles_small_display.pmtile',
        hires: 'display/hires_visual_tiles_small_display.pmtile'
    },
    micro: {
        radar: 'display/sar_tiles_micro_display.pmtile',
        lores: 'display/visual_tiles_micro_display.pmtile',
        hires: 'display/hires_visual_tiles_micro_display.pmtile'
    }
}

const presign_url = async key => {
    let req = await fetch(`${LAMBDA_BASE}/sign/${key}`)
    let body = await req.json()

    if ('url' in body) {
        return body.url
    }

    return null
}

window.presign_url = presign_url


const setup_map = () => {
    const map = L.map('map').setView(LOC_CENTER, 5)

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


    // L.marker(LOC_ATLANTA).addTo(map)
    //     .bindPopup('atlanta: 8/125/79')
    //     // .openPopup()

    return [map, tile_layer, debug_layer]
}

const read_map_controls = () => {
    const sel_dataset = document.querySelector('#mapdataset')
    const sel_bands = document.querySelector('#mapbands')
    const sel_grid = document.querySelector('#mapgrid')

    const dataset = !!sel_dataset.value ? sel_dataset.value : 'full'
    const bands = !!sel_bands.value ? sel_bands.value : 'radar'

    console.log(dataset)

    const tileset = DATASETS[dataset][bands]
    
    return {
        tileset,
        grid: sel_grid.value === 'on'
    }
}


window.addEventListener('load', () => {
    console.log('it begins')

    const [map, tile_layer, debug_layer] = setup_map()


    const update_map = async () => {
        const obj = read_map_controls()

        let tile_url = await presign_url(obj.tileset)

        debug_layer.removeFrom(map)
        tile_layer.clearLayers()

        const p = new PMTiles(tile_url)
        leafletRasterLayer(p).addTo(tile_layer)

        if (obj.grid) {
            debug_layer.addTo(map)
        }
    }


    update_map()

    document.querySelector('#mapshow').addEventListener('click', update_map)

})
