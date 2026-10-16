const HEADER = `<header><nav aria-label="Main"><a href="index.html">Home</a> <a href="products.html">Products</a> <a href="about.html">About</a> <a href="cart.html" id="cart-link">Cart (<span id="cart-count">0</span>)</a></nav><div class="account"><button id="account-btn" aria-haspopup="menu" aria-expanded="false" onclick="toggleMenu()">Account</button><div role="menu" id="account-menu" hidden><a role="menuitem" href="about.html">Help</a> <a role="menuitem" href="orders.html">My orders</a></div></div></header>`;
function toggleMenu() {
  const m = document.getElementById('account-menu'), b = document.getElementById('account-btn');
  const open = m.hidden; m.hidden = !open; b.setAttribute('aria-expanded', String(open));
}
function cartItems() { return JSON.parse(localStorage.getItem('cart') || '[]'); }
function updateCount() { const el = document.getElementById('cart-count'); if (el) el.textContent = cartItems().length; }
function addToCart(name) {
  const c = cartItems(); c.push(name); localStorage.setItem('cart', JSON.stringify(c)); updateCount();
  document.getElementById('added-name').textContent = name;
  document.getElementById('added-dialog').showModal();
}
document.addEventListener('DOMContentLoaded', () => {
  document.body.insertAdjacentHTML('afterbegin', HEADER);
  updateCount();
  const title = document.getElementById('product-title');
  if (title) {
    const id = new URLSearchParams(location.search).get('id') || '1';
    title.textContent = 'Product ' + id;
    document.getElementById('product-add').onclick = () => addToCart('Product ' + id);
  }
  const list = document.getElementById('cart-items');
  if (list) cartItems().forEach(n => { const li = document.createElement('li'); li.textContent = n; list.appendChild(li); });
  const r = document.getElementById('results-for');
  if (r) r.textContent = new URLSearchParams(location.search).get('q') || '';
});
