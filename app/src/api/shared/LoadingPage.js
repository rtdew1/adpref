import React from 'react';

export default class LoadingPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      img_fallback: "https://cdnjs.cloudflare.com/ajax/libs/galleriffic/2.0.1/css/loader.gif"
    }
  }

  render() {
    return (<div className="App">
        <header className="App-header">
          <div>
            <h2>Loading Page</h2>
            <img src={this.state.img_fallback} className="Loading" alt="Loading application" />
          </div>
        </header>
      </div>
    )
  }
}
