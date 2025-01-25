const path = require('path');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
    const isDevelopment = argv.mode === 'development';

    return {
        mode: isDevelopment ? 'development' : 'production',
        entry: './static/js/main.js',
        output: {
            filename: 'bundle.js',
            path: path.resolve(__dirname, 'main/static/main'),
            clean: true, // Очищает директорию вывода перед сборкой
        },
        module: {
            rules: [
                {
                    test: /\.css$/,
                    use: [
                        isDevelopment ? 'style-loader' : MiniCssExtractPlugin.loader,
                        'css-loader',
                    ],
                },
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env'],
                        },
                    },
                },
            ],
        },
        plugins: [
            new MiniCssExtractPlugin({
                filename: 'bundle.css',
            }),
        ],
        devServer: {
            static: path.join(__dirname, 'main/static/main'),
            compress: true,
            port: 9000,
            hot: true,
            proxy: {
                '/': {
                    target: 'http://127.0.0.1:8000',
                    secure: false,
                    changeOrigin: true,
                },
            },
        },
        devtool: isDevelopment ? 'inline-source-map' : false,
    };
};
