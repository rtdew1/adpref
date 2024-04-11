import React from 'react';
import { instanceOf } from 'prop-types';
import { withCookies, Cookies } from 'react-cookie';
import '../../App.css';

// Launch page for the app - button sends to Initial image ratings
class Home extends React.Component {

  static propTypes = {
    cookies: instanceOf(Cookies).isRequired
  };

  constructor(props) {
    super(props);
    this.state = {
      user: "N/A",
      img_fallback: "https://cdnjs.cloudflare.com/ajax/libs/galleriffic/2.0.1/css/loader.gif",
      version: null
    }
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  // React function that is called when the page loads.
  async componentDidMount() {
    const { cookies } = this.props;
    console.log(cookies.get('user'));

    const version_res = await fetch(
      process.env.REACT_APP_BACKEND_ENDPOINT + "/version",
      { method: "GET" }
    )
    const version = (await version_res.json())["version"]
    this.setState({
      version: version
    })

    if (cookies.get('user') === "undefined" || cookies.get('user') === undefined) {
      // Send an HTTP request to the server.
      await fetch(
        process.env.REACT_APP_BACKEND_ENDPOINT + '/create_user', { method: 'GET' } // The type of HTTP request.
      ).then
        (res => {
          return res.json();
        },
          err => {
            // Print the error if there is one.
            console.log(err);
          }).then(
            data => {
              cookies.set('user', data.user_id, { path: '/', maxAge: 3600 })
              this.setState({
                user: data.user_id
              });
            },
            err => {
              // Print the error if there is one.
              console.log(err);
            }
          )
        ;
    }
  }

  // Send the mturk ID to the backend and move to survey
  async handleSubmit(event) {
    event.preventDefault();
    const data = new FormData(event.target);

    fetch(process.env.REACT_APP_BACKEND_ENDPOINT + '/mturk_receive/' + this.state.user, {
      method: 'POST',
      body: data,
    }).then(
      data => { window.location.href = "/initial" },
      err => {
        console.log(err);
        window.location.href = "/initial"
      }
    );
  }

  render() {
    let page_body;

    if (this.state.user === "N/A") {
      page_body = (
        <div>
          <h2>Loading</h2>
          <img src={this.state.img_fallback} className="Loading" alt="Loading application" />
        </div>
      )
    } else {
      page_body = (
        <div>
          <div className='App-version'>
            App version: {this.state.version} <br />
            Frontend version: ar-1.0
          </div>
          <p>In the following questions, we will try to measure your <br />
            preferences over women's tops and dresses.
          </p>
          <p>
            We'll start by showing you a series of random items, and ask <br />
            you to rate them from 0 to 10, where 10 is the best. Your rating <br />
            should be derived from the image: do I like this dress, based on <br />
            the photo? Decimals, like 7.5, are fine.
          </p>
          <p>It may take a few moments for the newest dress <br />
            to load. Please be patient and do not hit the refresh or back buttons.
          </p>
          <p>
            Enter your MTurk ID below to begin. (If you do not have one, <br />
            please enter your name, first and last, instead).
          </p>
          <form onSubmit={this.handleSubmit}>
            <input id="mturk_id" name="mturk_id" required />
            <button className="App-button">Submit</button>
            <br />
          </form>
        </div>
      )
    }

    return (
      <div className="App">
        <header className="App-header">
          {page_body}
        </header>
      </div>);
  }
}

export default withCookies(Home);