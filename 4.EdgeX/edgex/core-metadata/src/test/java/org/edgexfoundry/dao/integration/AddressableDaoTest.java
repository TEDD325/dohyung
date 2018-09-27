/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.AddressableData.TEST_ADDR_NAME;
import static org.edgexfoundry.test.data.AddressableData.checkTestData;
import static org.edgexfoundry.test.data.AddressableData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Asset;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class AddressableDaoTest {

  @Autowired
  private AddressableDao dao;

  @Autowired
  private AddressableRepository repos;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  private String addressableId;

  @Before
  public void creatTestData() {
    Addressable addr = newTestInstance();
    repos.save(addr);
    addressableId = addr.getId();
    assertNotNull("new test Addressable has no identifier", addressableId);
    assertNotNull("Dao is null", dao);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
    deviceRepos.deleteAll();
    serviceRepos.deleteAll();
  }

  @Test
  public void testGetById() {
    Addressable addr = dao.getById(addressableId);
    assertNotNull("Addressable is null on getById", addr);
    checkTestData(addr, addressableId);
  }

  @Test
  public void testGetByIdWithBadId() {
    assertNull("No addressable should be found with bad id", dao.getById("badid"));
  }

  @Test
  public void testGetByName() {
    Addressable addr = dao.getByName(TEST_ADDR_NAME);
    assertNotNull("Addressable is null on getByName", addr);
    checkTestData(addr, addressableId);
  }

  @Test
  public void testGetByNameWithBadName() {
    assertNull("No addressable should be found with bad name", dao.getByName("badname"));
  }

  @Test
  public void testGetByIdOrName() {
    Addressable addr = newTestInstance();
    assertNotNull("Addressable is null on getByIdOrName with valid id", dao.getByIdOrName(addr));
    addr.setName(null);
    addr.setId(addressableId);
    assertNotNull("Addressable is null on getByIdOrName with valid name", dao.getByIdOrName(addr));
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("No addressable should be found with null on getByIdOrName",
        dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrNameWithBadIdentifiers() {
    Addressable addr = newTestInstance();
    addr.setId("badid");
    assertNull("No addressable should be found with bad id on getByIdOrName", dao.getByIdOrName(addr));
    addr.setId(null);
    addr.setName("badname");
    assertNull("No addressable should be found with bad id on getByIdOrName", dao.getByIdOrName(addr));
  }

  @Test
  public void testIsAddressableAssociatedToDevice() {
    Device device = new Device();
    Addressable addr = repos.findOne(addressableId);
    device.setAddressable(addr);
    deviceRepos.save(device);
    assertTrue("Found no associated Devices and should have",
        dao.isAddressableAssociatedToDevice(addr));
  }

  @Test
  public void testIsAddressableAssociatedToDeviceWithNone() {
    assertFalse("No devices should have been found for the addressable",
        dao.isAddressableAssociatedToDevice(repos.findOne(addressableId)));
  }

  @Test
  public void testIsAddressableAssociateToDeviceService() {
    DeviceService service = new DeviceService();
    Addressable addr = repos.findOne(addressableId);
    service.setAddressable(addr);
    serviceRepos.save(service);
    assertTrue("Found no associated DeviceServices and should have one",
        dao.isAddressableAssociatedToDeviceService(addr));
  }

  @Test
  public void testIsAddressableAssociateToDeviceServiceWithNone() {
    assertFalse("No devices should have been found for the addressable",
        dao.isAddressableAssociatedToDevice(repos.findOne(addressableId)));
  }

  @Test
  public void testGetOwningServices() {
    DeviceService service = new DeviceService();
    serviceRepos.save(service);
    Addressable addr = repos.findOne(addressableId);
    Device device = new Device();
    device.setAddressable(addr);
    device.setService(service);
    deviceRepos.save(device);
    List<Asset> services = dao.getOwningServices(addr);
    assertEquals("Did not find correct number of associated DeviceServices ", 1, services.size());
    assertEquals("Associated service is not the same", service.getId(),
        ((DeviceService) services.get(0)).getId());
  }

  @Test
  public void testGetOwningServicesHavingNone() {
    assertTrue("Should be no owning device services",
        dao.getOwningServices(repos.findOne(addressableId)).isEmpty());
  }

  @Test
  public void testGetOwningServicesWithNull() {
    assertTrue("Should be no owning device services", dao.getOwningServices(null).isEmpty());
  }

}
