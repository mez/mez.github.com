$('.contact-link').each(function(i, link) {
  $(link).prop('href', 'mail' + 'to' + ':mez' + '@' + 'jetcode' + '.io');
});
$('.mobile-toggle').on('click', function(e) {
  e.preventDefault();
  $('.toggle-content').toggleClass('open');
});
