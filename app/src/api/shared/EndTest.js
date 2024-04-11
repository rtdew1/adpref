import React from 'react';
import { instanceOf } from 'prop-types';
import { withCookies, Cookies } from 'react-cookie';
import '../../App.css';

// Closes out testing ratings and shows recommendations for user
class EndTest extends React.Component {

  static propTypes = {
    cookies: instanceOf(Cookies).isRequired
  };

  constructor(props) {
    super(props);
    const { cookies } = props;
    this.state = {
      recs: [],
      user: cookies.get('user'),
      imageUrls: []
    }
  }

  renderImage(imageUrl, i) {
    return (
      <tr key={"row-" + (i + 1)}>
        <td>
          <img src={imageUrl} className="App-photo" alt={"Recommended dress " + (i + 1)}>
          </img>
        </td>
      </tr>
    );
  }

  // React function that is called when the page loads.
  componentDidMount() {
    // Send an HTTP request to the server.
    fetch(
      process.env.REACT_APP_BACKEND_ENDPOINT + '/load_recs/' + this.state.user, { method: 'GET' } // The type of HTTP request.
    ).then(
      res => {
        // Convert the response data to a JSON.
        return res.json();
      }, err => {
        // Print the error if there is one.
        console.log(err);
      }
    ).then
      (data => {
        // Load the current user's recommendations
        console.log(data)

        this.setState({
          // If img_urls does not exist it must be replaced with an empty list.
          imageUrls: data.hasOwnProperty('img_urls') ? data.img_urls : [],
          shortid: data.mturk_id
        });
      },
        err => {
          // Print the error if there is one.
          console.log(err);
        });
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <p>
            Thanks for taking our survey! <br />
            Please enter this code on the next page of the Qualtrics survey: <br />
            {this.state.user} <br />
          </p>

          <p>
            Note: this is NOT the code you should submit on Mturk. <br />
            We will give you a different code at the end of the Qualtrics survey for completing the HIT.
          </p>
        </header>
      </div>
    );
  }
}

export default withCookies(EndTest);

// Code for showing recommendations
// <table>
//   <thead>
//   </thead>
//   <tbody>
//     {this.state.imageUrls.map((imageUrl, i) => this.renderImage(imageUrl["img_url"], i))}
//   </tbody>
// </table>
