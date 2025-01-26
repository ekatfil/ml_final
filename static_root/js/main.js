import '../css/styles.css';

// Динамический импорт Bootstrap
import(/* webpackChunkName: "bootstrap" */ 'bootstrap/dist/js/bootstrap.bundle.min.js').then((module) => {
    // Bootstrap загружен
});
